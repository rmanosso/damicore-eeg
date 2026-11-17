# DAMICORE v2 — EEG: Controle × CCL (MCI) × Alzheimer (AD)

Pasta independente para a apresentação: contém o notebook, os dados e as saídas.

```
damicore_v2.ipynb    notebook, construído passo a passo
dados/               dataset original (AD/, CONTROL/, MCI/, readme.txt); baixado pelo notebook se faltar
saida/               gerado pelo notebook
├── pacientes.csv
├── objetos/<estado>/<canal>/<paciente>.bin          passo 1
├── damicore/<estado>/k2/<canal>/                    passo 2: artefatos do DAMICORE
├── discretizacao/tabela_<estado>_k2.csv              passo 2: pacientes x canais, valores 0/1
└── grupos_canais/<estado>/                          passo 3: DAMICORE sobre a tabela (canais como objetos)
    ├── entrada.csv
    └── damicore/                                    matriz de distâncias, filograma, grupos, relatório
```

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook damicore_v2.ipynb
```

## Dados

Cejnek, M.; Vysata, O.; Valis, M.; Bukovsky, I. *Novelty detection-based approach for Alzheimer's disease and mild cognitive
impairment diagnosis from EEG*. Med. Biol. Eng. Comput. 59, 2287–2296 (2021). [doi:10.1007/s11517-021-02427-6](https://doi.org/10.1007/s11517-021-02427-6)

Dataset: [doi:10.6084/m9.figshare.5450293](https://doi.org/10.6084/m9.figshare.5450293), licença CC BY 4.0.
