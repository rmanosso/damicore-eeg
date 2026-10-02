# DAMICORE em EEG: Controle × CCL × Alzheimer

Aplicação do [DAMICORE](https://github.com/Delbem-Research-and-Innovation/damicore) a sinais de EEG em repouso
de pacientes saudáveis (controle), com comprometimento cognitivo leve (CCL/MCI) e com doença de Alzheimer (AD).
O objetivo é uma primeira análise dos dados sem especialista: os sinais entram quase crus, e o diagnóstico só é usado
no fim, para avaliar os agrupamentos.

Trabalho da disciplina de pós-graduação sobre Algoritmos de Estimação de Distribuição de propósito geral (Prof. Alexandre Delbem, ICMC-USP).

## Pipeline

Tudo está em [`scripts/script-alzheimer.ipynb`](scripts/script-alzheimer.ipynb):

1. **Inventário e harmonização dos sinais**: reamostragem para 128 Hz, passa-banda comum de 4,5–40 Hz, janela de 48 s,
   z-score por canal e quantização em 8 níveis. Sem essa harmonização, os grupos refletiriam o aparelho de registro, não a doença.
2. **Etapa 1 — DAMICORE por canal**: uma execução para cada um dos 19 canais. Os objetos são os pacientes (um arquivo por paciente).
   O cluster de cada paciente em cada canal é a discretização do canal. `NUM_CLUSTERS` define k (`None` = automático).
3. **Tabelas de probabilidade** a partir da tabela discretizada (paciente × canal).
4. **Etapa 2 — DAMICORE entre pacientes**: a tabela de probabilidade marginal, já transposta (canais × pacientes),
   é agrupada novamente numa única chamada, `etapa2(k=...)`.

## Como executar

### Local

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt jupyter
```

Baixe os dados (seção abaixo), extraia em `dataset/dataset/` e execute o notebook de cima para baixo.

### Google Colab

Abra o notebook pelo GitHub no Colab e descomente as linhas da célula **"Ambiente e dados"**. Elas instalam o DAMICORE,
clonam este repositório e baixam os dados originais. A primeira execução completa leva cerca de 15 min
(o DAMICORE leva ~47 s por canal); as seguintes reaproveitam os resultados já calculados.

## Dados

Cejnek, M.; Vysata, O.; Valis, M.; Bukovsky, I. *Novelty detection-based approach for Alzheimer's disease and mild cognitive
impairment diagnosis from EEG*. Medical & Biological Engineering & Computing 59, 2287–2296 (2021).
[doi:10.1007/s11517-021-02427-6](https://doi.org/10.1007/s11517-021-02427-6)

Dataset: [doi:10.6084/m9.figshare.5450293](https://doi.org/10.6084/m9.figshare.5450293), licença CC BY 4.0.
Contém 59 AD, 7 MCI e 102 controles, com EEG de 19 canais (sistema 10-20) registrado em dois aparelhos (256 Hz e 128 Hz).

```bash
mkdir -p dataset/dataset
wget -O dataset/dataset.zip https://ndownloader.figshare.com/files/9423433
unzip dataset/dataset.zip -d dataset/dataset
```

Os dados não fazem parte deste repositório.

## Estrutura

```
scripts/script-alzheimer.ipynb   notebook com todo o pipeline
dataset/dataset/                 dados originais (não versionados)
resultados/                      gerado pelo notebook (não versionado)
├── tabela_discretizada_<k>.csv  etapa 1: paciente × canal
├── filogramas/                  um filograma por canal
├── damicore/<parâmetros>/<canal>/   artefatos do DAMICORE por canal
└── etapa2/<etapa1>__<canais>__k<k>__c<casas>/   tabela transposta, clusters e avaliação da etapa 2
```

## Observações

- **Viés de aparelho:** cada aparelho chega com filtros diferentes, e quase todos os controles vêm de um único formato.
  O notebook mede esse viés (`NMI_equip_AD`) em cada resultado.
- **Rótulos dos clusters:** o número de um cluster não tem o mesmo significado de um canal para outro (cada canal é uma execução independente).
  A tabela marginal da etapa 2 não depende desses rótulos.
- **Sensibilidade da etapa 2:** cada paciente vira um objeto de ~130 bytes, e o NCD fica instável com objetos tão pequenos.
  Detalhes como o número de casas decimais mudam as contagens. A separação entre AD e controles se mantém, mas os números exatos variam.

## Referências

- Sanches, A.; Cardoso, J. M. P.; Delbem, A. C. B. *Identifying Merge-Beneficial Software Kernels for Hardware Implementation* (artigo que apresenta o DAMICORE).
- Gaspar-Cunha, A.; Costa, P.; Monaco, F.; Delbem, A. *Many-Objectives Optimization: A Machine Learning Approach for Reducing the Number of Objectives*.
  Math. Comput. Appl. 28, 17 (2023). [doi:10.3390/mca28010017](https://doi.org/10.3390/mca28010017)
- Gaspar-Cunha, A.; Costa, P.; Delbem, A.; Monaco, F.; Ferreira, M. J.; Covas, J. *Evolutionary Multi-Objective Optimization of Extrusion Barrier Screws:
  Data Mining and Decision Making*. Polymers 15, 2212 (2023). [doi:10.3390/polym15092212](https://doi.org/10.3390/polym15092212)
