# Teste dos compressores do DAMICORE 0.5.0 nos objetos do V4 (só leitura de dados/ e saida/pacientes.csv).
#
# Rodar com o ambiente da 0.5.0, a partir da pasta DAMICORE_v4:
#   .venv05/Scripts/python testes/teste_compressores.py
#
# Para cada pasta de equipamento, canal de teste e compressor, mede:
#   validade      ncd_self_largest (o compressor reconhece um objeto repetido?) e pares acima da janela
#   escala do NCD mediana e amplitude interquartil das distâncias entre pacientes (perto de 1 = nada em comum)
#   identificação duas janelas vizinhas, sem sobreposição, de cada paciente (A e B). O vizinho mais próximo
#                 de cada janela é a outra janela do mesmo paciente? E a AUC que separa as distâncias
#                 intra-paciente das entre pacientes. Mede se o NCD captura o que é próprio de cada paciente.
#   estabilidade  o corte K=2 da árvore feita só com as janelas A coincide com o das janelas B? (ARI)
#   custo         tempo da execução
# E uma checagem de regressão: zlib na 0.5.0 contra as distâncias que a 0.3.0 gravou no V4.
import json, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import adjusted_rand_score, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / 'saida' / 'teste_compressores'
CANAIS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
          'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
CANAIS_TESTE = ['Fz', 'C3', 'T4', 'Pz', 'O1']          # frontal, central, temporal, parietal, occipital
COMPRESSORES = ['zlib', 'ppmd', 'lzma', 'bzip2']
PASTAS = ['fir', 'export128', 'export256']
N = 3840


def ler_sinal(linha):
    caminho = RAIZ / linha.destino
    if linha.pasta == 'fir':
        return sio.loadmat(caminho)['segmenty'][:19].T.astype(np.float64)
    partes = [x for x in (sio.loadmat(f)['export'] for f in sorted(caminho.glob('*.mat'), key=lambda f: int(f.stem)))
              if x.shape[1] == 19]
    return np.concatenate(partes).astype(np.float64)


def preparar_objetos():
    org = pd.read_csv(RAIZ / 'dados' / 'organizacao.csv').set_index('paciente')
    pac = pd.read_csv(RAIZ / 'saida' / 'pacientes.csv').set_index('paciente')
    dir_obj = SAIDA / 'objetos'
    if dir_obj.exists():
        shutil.rmtree(dir_obj)
    usados, fora = [], []
    for pid in pac.index:
        x = ler_sinal(org.loc[pid])
        if len(x) < 2 * N:
            fora.append(dict(paciente=pid, pasta=pac.loc[pid, 'pasta'], amostras=len(x)))
            continue
        meio = len(x) // 2
        jan = {'A': x[meio - N:meio].astype(np.float32), 'B': x[meio:meio + N].astype(np.float32)}
        for canal in CANAIS_TESTE:
            i = CANAIS.index(canal)
            d = dir_obj / pac.loc[pid, 'pasta'] / canal
            d.mkdir(parents=True, exist_ok=True)
            for k, j in jan.items():
                (d / f'{pid}__{k}.bin').write_bytes(np.ascontiguousarray(j[:, i]).tobytes())
        usados.append(dict(paciente=pid, pasta=pac.loc[pid, 'pasta'], estado=pac.loc[pid, 'estado']))
    pd.DataFrame(fora).to_csv(SAIDA / 'fora_do_teste.csv', index=False)
    return pd.DataFrame(usados), pd.DataFrame(fora)


def corte_k2(dist, rotulos, destino):
    from damicore_tree_builder import neighbor_joining
    from damicore_tree_builder.artifacts import write_tree_artifacts
    from damicore_clusterizer import cluster_tree, ClusterConfig
    destino.mkdir(parents=True, exist_ok=True)
    arv = neighbor_joining(dist, rotulos)
    tree_path, _ = write_tree_artifacts(arv, destino)
    res = cluster_tree(tree_path, destino / 'corte', config=ClusterConfig(num_clusters=2))
    m = pd.read_csv(res.membership_path)
    return pd.Series(m.cluster.values, index=m.label.values)


def medir(pasta, canal, comp):
    from damicore import run, ExecutionConfig, DamicoreError
    destino = SAIDA / 'execucoes' / pasta / canal / comp
    if destino.exists():
        shutil.rmtree(destino)
    t0 = time.time()
    for tentativa in range(5):
        try:
            r = run(SAIDA / 'objetos' / pasta / canal, source_kind='files', compressor=comp, num_clusters=None,
                    output_dir=destino, progress=False, execution=ExecutionConfig(workers='auto'))
            break
        except DamicoreError as e:              # Windows: arquivo temporário bloqueado; retoma do checkpoint
            if 'PermissionError' not in str(e) or tentativa == 4:
                raise
            time.sleep(2)
    r.close()
    tempo = time.time() - t0
    rel = json.loads((destino / 'report.json').read_text(encoding='utf-8'))
    D = np.load(destino / 'distance.npy')
    rot = [l.removesuffix('.bin') for l in json.loads((destino / 'labels.json').read_text(encoding='utf-8'))['labels']]
    pid = np.array([r.split('__')[0] for r in rot])
    jan = np.array([r.split('__')[1] for r in rot])
    # identificação: vizinho mais próximo de cada janela
    M = D.copy()
    np.fill_diagonal(M, np.inf)
    viz = M.argmin(1)
    top1 = float(np.mean(pid[viz] == pid))
    iu = np.triu_indices(len(rot), 1)
    intra = pid[iu[0]] == pid[iu[1]]
    d = D[iu]
    auc = float(roc_auc_score(intra, -d))
    inter = d[~intra]
    # estabilidade: corte K=2 com só as janelas A contra só as janelas B
    part = {}
    for k in ('A', 'B'):
        idx = np.where(jan == k)[0]
        part[k] = corte_k2(D[np.ix_(idx, idx)], list(pid[idx]), destino / f'so_{k}')
    comuns = part['A'].index
    ari = float(adjusted_rand_score(part['A'][comuns], part['B'].reindex(comuns)))
    return dict(pasta=pasta, canal=canal, compressor=comp, pacientes=len(set(pid)), tempo_s=round(tempo, 1),
                ncd_self_largest=rel.get('ncd_self_largest'), pares_acima_janela=rel.get('pairs_over_window'),
                ncd_inter_mediana=float(np.median(inter)), ncd_inter_iqr=float(np.subtract(*np.percentile(inter, [75, 25]))),
                ncd_intra_mediana=float(np.median(d[intra])), ncd_inter_min=float(inter.min()), ncd_inter_max=float(inter.max()),
                ident_top1=top1, acaso_top1=1 / (len(rot) - 1), auc_intra_inter=auc, estabilidade_k2_ari=ari,
                k_automatico=rel['cluster_count'])


def regressao():
    # zlib 0.5.0 nos objetos do V4 (janela central) contra a matriz da 0.3.0
    from damicore import run, ExecutionConfig
    linhas = []
    for pasta in PASTAS:
        canal = 'Fz'
        v4 = RAIZ / 'saida' / pasta / 'damicore_canais' / canal
        destino = SAIDA / 'regressao' / pasta
        if destino.exists():
            shutil.rmtree(destino)
        r = run(RAIZ / 'saida' / pasta / 'objetos' / canal, source_kind='files', compressor='zlib', num_clusters=None,
                output_dir=destino, progress=False, execution=ExecutionConfig(workers='auto'))
        r.close()
        rot_a = json.loads((v4 / 'labels.json').read_text(encoding='utf-8'))['labels']
        rot_b = json.loads((destino / 'labels.json').read_text(encoding='utf-8'))['labels']
        a = pd.DataFrame(np.load(v4 / 'distance.npy'), rot_a, rot_a)
        b = pd.DataFrame(np.load(destino / 'distance.npy'), rot_b, rot_b).loc[rot_a, rot_a]
        ma = pd.read_csv(v4 / 'membership.csv').set_index('label').cluster
        mb = pd.read_csv(destino / 'membership.csv').set_index('label').cluster
        linhas.append(dict(pasta=pasta, canal=canal, max_dif_ncd=float((a - b).abs().values.max()),
                           mesmos_grupos_auto=bool(adjusted_rand_score(ma, mb.reindex(ma.index)) == 1.0)))
    return pd.DataFrame(linhas)


def main():
    import damicore
    SAIDA.mkdir(parents=True, exist_ok=True)
    print('damicore', getattr(damicore, '__version__', '?'))
    arq = SAIDA / 'resultados.csv'
    linhas = pd.read_csv(arq).to_dict('records') if arq.is_file() else []   # retoma uma execução interrompida
    feitos = {(l['pasta'], l['canal'], l['compressor']) for l in linhas}
    if not (SAIDA / 'objetos').exists() or not linhas:
        usados, fora = preparar_objetos()
        print(f'{len(usados)} pacientes com duas janelas; {len(fora)} fora do teste (sinal < {2 * N} amostras)')
        print(usados.groupby('pasta').size().to_dict())
    if not (SAIDA / 'regressao_zlib.csv').is_file():
        reg = regressao()
        reg.to_csv(SAIDA / 'regressao_zlib.csv', index=False)
        print(reg)
    for pasta in PASTAS:
        for canal in CANAIS_TESTE:
            for comp in COMPRESSORES:
                if (pasta, canal, comp) in feitos:
                    continue
                lin = medir(pasta, canal, comp)
                linhas.append(lin)
                pd.DataFrame(linhas).to_csv(arq, index=False)
                print(f"{pasta:>9} {canal:>3} {comp:>5}: top1 {lin['ident_top1']:.2f}  AUC {lin['auc_intra_inter']:.2f}  "
                      f"ARI {lin['estabilidade_k2_ari']:.2f}  NCD {lin['ncd_inter_mediana']:.3f}±{lin['ncd_inter_iqr']:.3f}  "
                      f"self {lin['ncd_self_largest']:.3f}  {lin['tempo_s']:.0f}s", flush=True)
    print('ok')


if __name__ == '__main__':
    main()
