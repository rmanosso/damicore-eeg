# DAMICORE v4: um universo por equipamento

No V3, com todos os pacientes juntos, a discretização separou os pacientes pelo aparelho quase tanto quanto pelo estado
(quase todos os controles são arquivos `fir`, todos os AD são `export`). No V4, cada equipamento é processado em separado,
do começo ao fim:

| Pasta | Arquivos | Taxa | Estados |
|---|---|---|---|
| `fir` | `*fir.mat` (inteiros) | 128 Hz | CONTROL |
| `export128` | `export` | 128 Hz | CONTROL (normal1, normal2), MCI, AD |
| `export256` | `export` | 256 Hz | MCI, AD |

Em cada pasta: objetos (sinais originais, sem reamostrar e sem filtrar) → discretização (DAMICORE por canal, corte em K grupos) →
grupos de canais (DAMICORE com os canais como objetos) → tabelas de probabilidade. Um modelo por pasta, com todos os pacientes;
o estado é só rótulo nas figuras. Não há classificação: dentro de uma pasta não existe contraste de estado.

## Como rodar

O V4 usa o **DAMICORE 0.5.0**, num ambiente próprio (`.venv05`), separado do `EEG/.venv` (0.3.0) usado pelo V3. Para recriá-lo:

```
python -m venv .venv05
.venv05\Scripts\python -m pip install -r requirements.txt ipykernel
.venv05\Scripts\python -m ipykernel install --user --name damicore05 --display-name "DAMICORE 0.5.0 (V4)"
```

1. Os brutos ficam em `c:\DAMICORE\dataset\dataset` (pastas `CONTROL`, `AD`, `MCI` e `readme.txt`). Se não existirem, o notebook baixa do figshare.
2. Abra `damicore_v4.ipynb` no kernel **DAMICORE 0.5.0 (V4)** e execute tudo.
   Na primeira vez, os dados são copiados para `dados/`, já separados por equipamento, e o DAMICORE roda 57 vezes (19 canais × 3 pastas).
   As execuções seguintes reaproveitam o que já foi calculado.

## Onde mexer

- **Compressor (seção 0.2):** `COMPRESSOR = {'fir': 'zlib', 'export128': 'ppmd', 'export256': 'ppmd'}`, escolhido pelo teste
  `testes/teste_compressores.py` (rodar com `.venv05\Scripts\python testes\teste_compressores.py`; cerca de 30 minutos).
  O passo 3 usa `COMPRESSOR_ESTRUTURA = 'zlib'`.
- **Caixa de discretização (seção 0.1):** `K_DISCRETIZACAO = {'fir': 2, 'export128': 2, 'export256': 2}`. Um K por pasta, padrão 2 (símbolos 0 e 1).
- **Painel (seção 2.4):** escolha a pasta e o canal, veja a árvore cortada e ajuste o K de um canal específico. Os ajustes vão para `saida/k_ajustes.json`.
- **Decisões (`decisoes.yaml`):** o registro de cada decisão do processo. Edite para acrescentar uma decisão e execute de novo o passo 6.

## Saídas

```
dados/                      cópia dos brutos, separada por equipamento (organizacao.csv diz de onde veio cada arquivo)
saida/
├── pacientes.csv, pacientes_retirados.csv, k_ajustes.json
├── <pasta>/objetos/        um .bin por paciente em cada pasta de canal
├── <pasta>/damicore_canais/  NCD, árvore e grupos de cada canal
├── <pasta>/<configuração>/   tabela_discretizacao.csv, grupos_canais/, probabilidades/
└── relatorios/
    ├── relatorio_<configuração>.html   relatório da execução
    ├── metricas_<configuração>.json
    └── decisoes.html                   árvore de decisões com o caminho tomado
```

## Gerador do notebook

`gerador/gerar_v4.py` gera o `damicore_v4.ipynb` (sem saídas). Para mudar o notebook, edite o gerador e rode, de dentro desta pasta,
`.venv05\Scripts\python gerador/gerar_v4.py damicore_v4.ipynb`; depois execute o notebook no kernel **DAMICORE 0.5.0 (V4)**.
