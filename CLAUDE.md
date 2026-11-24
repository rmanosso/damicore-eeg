# DAMICORE em EEG: contexto do projeto

Documento de passagem entre conversas (e leitura para o orientador). Resume o que foi construído até 2026-11-24,
os números que sustentam as decisões e o ponto de partida de cada uma das três vertentes de trabalho.
Responder sempre em português.

## O projeto

Aplicação do DAMICORE (NCD → árvore Neighbor Joining → FastGreedy) a EEG em repouso de controles, MCI e AD
(Cejnek et al. 2021; figshare 5450293), na disciplina de Algoritmos de Estimação de Distribuição (Prof. Alexandre Delbem, ICMC-USP).
O DAMICORE é usado como ciclo de um EDA: discretizar os sinais, aprender a estrutura (grupos de canais = blocos) e estimar
tabelas de probabilidade (modelo de produto marginal, como no ECGA).

## Estrutura

| Pasta | O que é |
|---|---|
| `dataset/dataset/` | brutos (CONTROL, AD, MCI, readme.txt); ignorado pelo git; fonte única para todas as versões |
| `scripts/` | primeiros notebooks (v1); o `README.md` da raiz ainda descreve só eles |
| `DAMICORE_v2/` | passo a passo com mais controle |
| `DAMICORE_v3/` | ciclo completo do EDA com sinais crus e viés do aparelho (encerrado) |
| `DAMICORE_v4/` | um universo por equipamento; modelo descritivo por pasta (atual) |
| `DAMICORE_ICA/` | novo projeto em outros dados (vertente 3) |

Cada versão é independente: `dados/` (cópia organizada dos brutos) e `saida/` são gerados e ignorados pelo git.
Os notebooks do V3 e do V4 são gerados por `gerador/gerar_v3.py` e `gerador/gerar_v4.py`: para mudar um notebook, edite o gerador e rode
`python gerador/gerar_v4.py damicore_v4.ipynb` (de dentro da pasta da versão), depois execute o notebook.

## Ambientes

| Ambiente | damicore | Kernel | Usado por |
|---|---|---|---|
| `C:\Users\rodri\Documents\EEG\.venv` | 0.3.0 | `eeg` | V2, V3 |
| `DAMICORE_v4\.venv05` | 0.5.0 | `damicore05` | V4 |

Execução sem abrir o Jupyter: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=<kernel> --ExecutePreprocessor.timeout=-1 <notebook>`.

Cuidados no Windows: scripts que chamam o DAMICORE precisam de `if __name__ == '__main__':` (multiprocessamento);
o DAMICORE às vezes falha com `PermissionError` num arquivo temporário (repetir a chamada retoma do checkpoint);
use `PYTHONIOENCODING=utf-8` para imprimir acentos e símbolos.

## Como trabalhar neste projeto

- Antes de implementar algo novo, apresentar o entendimento e as dúvidas e esperar a confirmação.
- Toda decisão vai para `DAMICORE_v4/decisoes.yaml` (pergunta, opções, escolhida, quem, quando, motivo, evidência, implicações).
  O notebook gera `saida/relatorios/decisoes.html` com a árvore de decisões.
- Sem orientação sobre uma escolha técnica: testar as opções com critérios definidos antes dos resultados e registrar.

## Achados e números-chave

**Viés do aparelho (V3).** Quase todos os controles são arquivos `fir` (inteiros, 128 Hz, 30 s) e todos os AD são `export`
(float, 128 ou 256 Hz). Com os sinais crus e todos juntos, a discretização segue o formato: NMI com o estado 0,598, com o formato 0,641;
os dois controles `export` (normal1, normal2) caem no grupo dos AD em 35 de 38 casos. A classificação de 155/157 não distingue doença de aparelho.

| | fir (controles) | export 128 Hz (AD) | export 256 Hz (AD) |
|---|---|---|---|
| valores | inteiros, passo 1 | float32 | float32 |
| compressão zlib da janela | ~29% | 88–93% | 88–93% |
| correlação média entre canais | +0,35 | −0,04 (referência média?) | −0,05 |
| potência no delta (1–4 Hz) | 25% | 16% | 0,1% |

Harmonizar (128 Hz, 4,5–40 Hz, z-score, 8 níveis) iguala taxa, filtros e escala, mas não a referência, e corta o delta (parte da
lentificação no AD). Decisão do usuário: não harmonizar e reportar o viés.

**V4: um universo por equipamento.** fir: 98 CONTROL; export128: 2 CONTROL, 3 MCI, 24 AD; export256: 4 MCI, 33 AD
(retirados: 26887fir e 7147fir por Cz plano; AD33 e AD44 sem segmento de 19 canais). Objetos de 3840 amostras float32 (15 KB; o par cabe
na janela de 32 KB do zlib). Sinais originais, sem reamostrar nem filtrar. Um modelo por pasta, estados só como rótulo; vai até as tabelas
de probabilidade (sem classificação, porque não há contraste de estado dentro das pastas). Com K=2: 6 grupos de canais em cada pasta
(o maior com 4 ou 5 canais); ocupação média das tabelas fir 95%, export128 56%, export256 72%. ARI entre os grupos de canais das pastas
entre −0,05 e 0,10 (estruturas diferentes).

**Teste de compressores (`DAMICORE_v4/testes/teste_compressores.py`).** Critérios em ordem: validade (NCD(x,x) ≈ 0), identificação do
mesmo paciente em duas janelas, estabilidade do corte K=2, custo. bzip2 inválido em todas as pastas; ppmd inválido na fir e melhor nas
export; na fir, zlib. Escolha: zlib na fir, ppmd nas export. Números em `decisoes.yaml`. Observações: o NCD entre pacientes fica perto de 1
com qualquer compressor (0,93–1,00), a identificação é fraca (melhor caso 24% contra 2% do acaso), e as árvores das pastas export saem em
"escada" mesmo com ppmd.

## Vertente 1 — consenso das árvores de canais (continua do V4)

A árvore de pacientes serve só para discretizar as séries temporais; o resultado de cada universo é a árvore de canais (passo 3). As três
árvores de canais têm as mesmas 19 folhas, então o consenso é aplicável a elas. Entre pacientes não é: nenhum paciente está em duas pastas
(nem supertree funciona com sobreposição zero).

- Tipos: estrito (bipartições em todas as árvores), maioria (>50%; com 3 árvores, 2 de 3), maioria estendida/guloso; comprimentos de ramo
  como média nas árvores que contêm a bipartição; ou média no espaço de árvores (Billera–Holmes–Vogtmann).
- Leitura: bipartição nas 3 pastas = estrutura independente de aparelho e estado (candidata: vizinhança dos eletrodos); nas duas export e
  não na fir = AD **ou** formato export (o confundimento volta entre pastas).
- Cuidados: só 3 árvores (suporte 1/3, 2/3, 3/3); árvores de canais instáveis (objeto de 29 a 98 bytes); avaliar estabilidade por bootstrap
  de pacientes dentro de cada pasta; reinserir canais idênticos (o passo 3 os junta num objeto) para ter as 19 folhas; consenso sobre a árvore
  inteira, não sobre os grupos do FastGreedy. Com estrutura comum, as pastas teriam os mesmos grupos e tabelas próprias (os símbolos de
  pastas diferentes não têm o mesmo significado).
- **Pendente:** a forma do consenso será decidida na orientação. Não implementar antes.

## Vertente 2 — dimensionamento de população a partir do V4

Modelo da ruína do jogador (Harik, Cantú-Paz, Goldberg e Miller, 1999): n ≈ −ln α · 2^(k−1) · σ_BB · √(2m) / d.
α = probabilidade de errar a decisão de um bloco; k = ordem do bloco; 2^(k−1) = competidores num alfabeto binário (com K símbolos, K^(k−1));
m = número de blocos; d/σ_BB = sinal sobre ruído entre o melhor e o segundo melhor bloco.

Correspondência proposta com o V4: bloco = grupo de canais do passo 3 (k até 5, m = 6 por pasta); o K da discretização é o alfabeto;
n disponível = pacientes da pasta (98, 29, 37). A teoria explica a esparsidade observada (ocupação baixa nas pastas pequenas, piora com K e
com grupos maiores). Ponto a desenvolver: o que fazem o papel de d e σ_BB num uso que não otimiza uma função de custo, e o que o
n calculado diz sobre o tamanho de amostra necessário para estimar a estrutura e as tabelas.

## Vertente 3 — DAMICORE_ICA (outros dados)

Pasta `DAMICORE_ICA/` (contém "Script do paper Lopes2022"). Dados e objetivo ainda não apresentados: perguntar antes de propor.
Lições que se transferem: checar viés de aquisição antes de tudo; objetos do mesmo tamanho e com o par dentro da janela do compressor;
testar o compressor; registrar decisões desde o início.

## Restrições

- ManyAnalyst (`C:\Datasets\EEGMANYANALYSTS\Blinded_Dataset`): nunca criar nem modificar nada nessa pasta; não publicar nada com esses dados.
- Commitar só quando o usuário pedir; dados e saídas não vão para o git.
