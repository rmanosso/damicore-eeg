# DAMICORE v3 — EEG: Controle × Alzheimer (AD)

Versão didática para a disciplina: o ciclo completo de um Algoritmo de Estimação de Distribuição (EDA) aplicado à identificação
do estado a partir do EEG. O EDA aprende a distribuição dos dados de cada estado, gera indivíduos sintéticos a partir dela e usa
essas distribuições para classificar um indivíduo como CONTROL ou AD.

Pasta independente: contém o notebook, os dados e as saídas.

## Dois modos: harmonizado e crus

A célula `HARMONIZAR` (seção 0.1) escolhe o que entra no DAMICORE:

- **crus** (`False`, padrão desta versão): os valores originais dos arquivos;
- **harmonizado** (`True`): reamostragem para 128 Hz, passa-banda comum de 4,5–40 Hz, z-score e quantização em 8 níveis.

O V3 trabalha com os sinais crus e registra o **viés do aparelho** como resultado: 98 dos 100 controles estão num formato (`fir`)
e todos os AD em outro (`export`), e a discretização segue o formato. Os dois controles gravados no formato dos AD (`normal1`, `normal2`)
caem junto com os AD e são classificados como AD. O modo harmonizado fica como teste de controle desse viés.

## Fluxo

| Passo | O que faz | No EDA |
|---|---|---|
| 1. Objetos | Pacientes CONTROL e AD, harmonizados ou crus; uma pasta por canal, com todos os pacientes juntos | população |
| 2. Discretização | DAMICORE em cada canal (NCD → filograma NJ, k automático) e corte de cada árvore no K escolhido para o canal | codificação |
| 3. Grupos de canais | DAMICORE na tabela de cada estado, canais como objetos, número de grupos automático | estrutura |
| 4. Tabelas de probabilidade | Uma por grupo de canais e estado: P(x) = ∏ P_g(x_g) | parâmetros |
| 5. Sintéticos | `N_SINTETICOS` indivíduos por estado, sorteados das tabelas | amostragem |
| 6. Classificação | escore = log P_CONTROL(x) − log P_AD(x), com suavização de Laplace | avaliação |
| 7. Relatório | `saida/relatorios/relatorio_<modo>_<configuração>.html`, com tabela comparativa entre configurações | — |

## Escolha do K de cada canal (painel do passo 2.4)

O FastGreedy corta a árvore de um canal em K grupos removendo K−1 ramos, e os cortes são aninhados: o corte *n* leva a árvore a *n*+1 grupos.
O notebook mostra:

- um mapa canais × K com o NMI entre os grupos e o estado (2.2);
- um painel com a árvore de cada canal: os ramos coloridos pelos grupos, os ramos cortados em vermelho e numerados, e duas faixas
  ao lado das folhas (estado do paciente e grupo). A escala pode ser a topologia (padrão, em que os cortes ficam visíveis) ou o comprimento dos ramos.

O painel precisa do `ipywidgets` **no kernel do notebook** (no VS Code, o ambiente escolhido no canto superior direito). Se faltar,
a célula 2.4 avisa; instale com `%pip install ipywidgets` numa célula e reinicie o kernel. Sem o painel, os K podem ser definidos
na célula `K_MANUAL` (por exemplo, `{'F4': 3, 'P4': 4}`).

Escolha o canal, mova o K, passe pelos 19 canais e clique em **Aplicar os K selecionados**. Os valores vão para
`saida/<modo>/k_por_canal.json`; depois, use **Executar abaixo** a partir da célula 2.5. Sem o arquivo, todos os canais usam `K_PADRAO = 2`.
Uma configuração com K iguais grava em `K2/` (ou `K3/`...); com K diferentes, em `Kcanal_<código>/`.

Para rodar o teste de controle, mude `HARMONIZAR` (0.1) para `True` e execute o notebook de novo. Mudar só os K reaproveita o NCD e os filogramas dos canais.
Quando o outro modo já rodou com a mesma configuração (ou com K=2 em todos), o relatório compara o viés de aparelho canal a canal.

**Escopo desta versão:** sem MCI e sem validação cruzada. Esses pontos ficam para o V4 e estão registrados como ressalvas no relatório.

```
damicore_v3.ipynb      notebook
dados/                 dataset original; copiado do DAMICORE_v2/ ou baixado se faltar
saida/
├── pacientes.csv, pacientes_retirados.csv
├── <modo>/                                            harmonizado/ ou crus/
│   ├── objetos/<canal>/<paciente>.bin                 passo 1
│   ├── damicore_canais/<canal>/                       passo 2: NCD, filograma, k automático (independe de K)
│   ├── k_por_canal.json                               K escolhidos no painel (passo 2.4)
│   └── <configuração>/                                K2/, K3/... ou Kcanal_<código>/
│       ├── corte_arvores/<canal>/                     passo 2: corte no K do canal
│       ├── tabela_discretizacao.csv                   passo 2: pacientes x canais
│       ├── grupos_canais/<estado>/                    passo 3
│       ├── probabilidades/<estado>/grupo<g>_*.csv     passo 4
│       ├── sinteticos.csv                             passo 5
│       └── classificacao_reais.csv, classificacao_sinteticos.csv   passo 6
└── relatorios/relatorio_<modo>_<configuração>.html, metricas_<modo>_<configuração>.json   passo 7
```

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook damicore_v3.ipynb
```

A primeira execução de cada modo calcula o NCD entre os 157 pacientes em cada um dos 19 canais (cerca de 17 min com os sinais crus);
as seguintes reaproveitam o cálculo.

## Dados

Cejnek, M.; Vysata, O.; Valis, M.; Bukovsky, I. *Novelty detection-based approach for Alzheimer's disease and mild cognitive
impairment diagnosis from EEG*. Med. Biol. Eng. Comput. 59, 2287–2296 (2021). [doi:10.1007/s11517-021-02427-6](https://doi.org/10.1007/s11517-021-02427-6)

Dataset: [doi:10.6084/m9.figshare.5450293](https://doi.org/10.6084/m9.figshare.5450293), licença CC BY 4.0.

## Gerador do notebook

`gerador/gerar_v3.py` gera o `damicore_v3.ipynb` (sem saídas). Para mudar o notebook, edite o gerador e rode, de dentro desta pasta,
`python gerador/gerar_v3.py damicore_v3.ipynb`; depois execute o notebook no kernel `eeg` (damicore 0.3.0).
