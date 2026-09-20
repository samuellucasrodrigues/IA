---

# Lista #3 — Inteligência Artificial

**Curso:** Ciência da Computação  
**Disciplina:** Inteligência Artificial — Profa. Cristiane Neri Nobre  
**Aluno:** Samuel Lucas Rodrigues Vieira  
**Tema:** Padrão de sobrevivência no desastre do TITANIC

---

### Roteiro do notebook

| Etapa | Conteúdo |
|---|---|
| 1 | Leitura da base, análise exploratória e remoção de atributos |
| 2 | Imputação de ausentes: **KNN Imputer** × **MissForest** (IterativeImputer + ExtraTrees) |
| 3 | Balanceamento: **SMOTE** × **RandomUnderSampling** (somente no treino) |
| 4 | Otimização de hiperparâmetros: **Random Search** × **Otimização Bayesiana (Optuna/TPE)** |
| 5 | Modelagem: **Árvore de Decisão** × **Random Forest** |
| 6 | Avaliação, regras extraídas e comparações (itens 4.b a 4.f) |

> **Decisão metodológica central:** a partição treino/teste é feita **antes** da imputação e do
> balanceamento. O imputador é ajustado apenas no treino e o balanceamento nunca toca o teste —
> caso contrário haveria vazamento de informação e as métricas ficariam otimistas.

---
## Etapa 1 — Leitura da base e análise exploratória

### 1.1 Remoção de atributos

Mantido o mesmo critério da Lista 02:

| Atributo | Motivo da remoção |
|---|---|
| `boat`, `body` | **Vazamento de informação** — só existem *depois* do desfecho |
| `cabin` | 77,5% ausente e 186 valores distintos |
| `home.dest` | 43,1% ausente e 369 valores distintos |
| `name` | 1.307 valores distintos → identificador |
| `ticket` | 929 valores distintos → identificador |

Diferente da Lista 02, **`age` não é mais descartado**: os 263 valores ausentes (20,1%) serão
imputados, que é exatamente o objeto do item 1.a desta lista.

---
## Etapa 2 — Item 1.a: imputação de dados ausentes

A partição é feita **antes** da imputação. O imputador é ajustado (`fit`) apenas no
conjunto de treino e depois aplicado (`transform`) ao teste, para que nenhuma informação
do teste influencie os valores imputados.

Os dois métodos comparados:

* **KNN Imputer** — substitui o ausente pela média ponderada dos *k* vizinhos mais próximos
  (k = 5, peso pela distância). Como depende de distância, os atributos são normalizados
  com `MinMaxScaler` antes da imputação e revertidos em seguida.
* **MissForest** — imputação iterativa baseada em árvores. A biblioteca `missingpy` está
  descontinuada, então usa-se o `IterativeImputer` do scikit-learn com `ExtraTreesRegressor`,
  que é a implementação equivalente: cada atributo com ausentes vira alvo de uma regressão
  por florestas usando os demais atributos como preditores, repetida por várias iterações.

### 2.1 Impacto de cada método sobre a distribuição dos dados

### 2.2 Qual método foi adotado e por quê

Os dois métodos preservam a distribuição de `fare` (só 1 ausente no treino, efeito nulo).
A diferença está em `age`, com 190 ausentes no treino:

* O **KNN Imputer** encolhe o desvio-padrão em **2,78%** e tem distância de Wasserstein de
  **0,73** em relação à distribuição observada.
* O **MissForest** encolhe o desvio em **6,00%** e tem Wasserstein de **0,88**; os valores
  que ele gera ficam muito concentrados em torno de 28 anos (desvio de apenas 9,4 contra
  14,4 do original), o clássico efeito de *regressão à média* de imputadores baseados em
  predição condicional.
* Nenhum dos dois é rejeitado pelo teste KS a 5% (p = 0,051 e p = 0,058), ou seja, ambos são
  aceitáveis — mas o KNN distorce menos a variabilidade original.

**Método adotado: KNN Imputer**, por preservar melhor a dispersão e a forma da distribuição de
`age`. Ainda assim, as duas bases imputadas seguem para a modelagem, porque o item 4.d pede a
comparação do efeito de cada uma sobre o desempenho final dos modelos.

---
## Etapa 3 — Item 1.b: balanceamento de classes

A classe é desbalanceada na razão **1,62 : 1** (61,8% mortos contra 38,2% sobreviventes).
Não é um desbalanceamento severo, mas é suficiente para enviesar a árvore em favor da classe
majoritária. Técnicas comparadas:

* **SMOTE** — *oversampling*: cria exemplos sintéticos da classe minoritária interpolando
  cada exemplo com seus vizinhos mais próximos.
* **RandomUnderSampling** — *undersampling*: sorteia e descarta exemplos da classe
  majoritária até igualar as duas.

O balanceamento é aplicado **apenas no conjunto de treino**. O teste permanece com a
distribuição real da população, pois é ele que simula o mundo real.

---
## Etapa 4 — Itens 2.b e 3.b: otimização de hiperparâmetros

Otimizadores comparados:

1. **Random Search** (`RandomizedSearchCV`) — sorteia combinações independentes no espaço de busca.
2. **Otimização Bayesiana** (`Optuna`, sampler TPE) — constrói um modelo probabilístico do
   espaço e concentra as avaliações nas regiões promissoras.

Para a comparação ser justa, os dois recebem **60 avaliações**, o mesmo espaço de busca, a mesma
validação cruzada (5 dobras estratificadas) e a mesma métrica: **F1 da classe minoritária**
(sobreviventes), que é a mais adequada num problema desbalanceado.

O balanceamento entra **dentro de um `Pipeline` do imbalanced-learn**, de modo que o SMOTE é
refeito a cada dobra da validação cruzada, usando só os dados de treino daquela dobra. Se o
SMOTE fosse aplicado antes da validação cruzada, exemplos sintéticos derivados da dobra de
validação apareceriam no treino e o escore ficaria inflado.

A busca é feita na configuração de referência **KNN Imputer + SMOTE**; os hiperparâmetros
vencedores são depois reutilizados em todas as combinações do item 4.a.

---
## Etapa 5 — Itens 2.a, 3.a e 4.a: treinamento e avaliação das combinações

Para cada uma das **4 combinações** de imputação × balanceamento, treinam-se os dois modelos
com os hiperparâmetros vencedores (o otimizador de maior F1 na validação cruzada), e avaliam-se
sempre no **mesmo conjunto de teste**, que nunca foi imputado com informação do treino nem
balanceado.

---
## Etapa 6 — Item 2.d: regras extraídas da árvore otimizada

### 6.1 Versão interpretável da árvore

A árvore otimizada maximiza o F1, mas gera **29 regras** — e várias delas são fragmentos
absurdos como `age > 30,60 E age <= 31,50`, que cobrem 9 exemplos. Isso é memorização, não
padrão. Como o objetivo do enunciado é **encontrar o padrão de sobrevivência**, treina-se a
mesma árvore limitada a `max_depth = 3`, exatamente o critério usado na Lista 02.

### 6.2 Item 4.f: importância dos atributos e interpretabilidade

---
## Etapa 7 — Item 4: discussão dos resultados

### 4.b — Qual modelo generaliza melhor?

| Modelo | Acurácia | Precisão | Recall | F1 | AUC | Gap treino−teste |
|---|---|---|---|---|---|---|
| Árvore de Decisão | 0,8162 | 0,7519 | 0,7750 | 0,7629 | 0,8563 | **+0,0173** |
| Random Forest | **0,8340** | **0,7969** | 0,7583 | **0,7771** | **0,8854** | **−0,0274** |

O **Random Forest generaliza melhor**. Ele vence em acurácia (+1,8 p.p.), F1 (+1,4 p.p.) e
sobretudo em AUC (+2,9 p.p.), que mede a qualidade do ranqueamento independentemente do limiar.
O sinal do *gap* é o dado mais eloquente: a árvore tem desempenho **melhor no treino que no
teste** (gap positivo, sinal de ajuste aos dados vistos), enquanto o RF tem desempenho **melhor
no teste que no treino** (gap negativo). Isso acontece porque cada árvore do RF é treinada num
*bootstrap* e só enxerga um subconjunto dos atributos por divisão: individualmente elas erram,
mas os erros são pouco correlacionados e a votação os cancela. A árvore única não tem esse
mecanismo — depende inteiramente da poda imposta pelos hiperparâmetros.

Vale registrar, porém, que a diferença é pequena (1,4 p.p. de F1 em 393 exemplos de teste ≈ 5
passageiros). Numa base pequena e com sinal dominado por um único atributo (`sex`), o ganho do
*ensemble* é naturalmente limitado.

### 4.c — Comparação dos otimizadores

| Modelo | Otimizador | F1 (CV) | Tempo (s) | Tempo/avaliação (s) |
|---|---|---|---|---|
| Árvore de Decisão | Random Search | 0,7155 | 2,12 | 0,035 |
| Árvore de Decisão | **Optuna (TPE)** | **0,7254** | 2,86 | 0,048 |
| Random Forest | Random Search | 0,7261 | 86,19 | 1,436 |
| Random Forest | **Optuna (TPE)** | **0,7315** | 116,60 | 1,943 |

*(os tempos absolutos dependem da máquina; o que importa é a razão entre os dois otimizadores)*

Com o mesmo orçamento de 60 avaliações, a **otimização bayesiana venceu nos dois modelos**:
+0,0099 de F1 na árvore e +0,0054 no Random Forest, em ambos os casos gastando cerca de 35% mais
tempo que o Random Search.

Houve diferença relevante nos hiperparâmetros encontrados. Na árvore, o Random Search parou em
`gini`, `max_depth = 4` e `max_features = 'log2'` — ou seja, sorteou uma árvore que só olha 3 dos
9 atributos em cada divisão, o que é ruim num problema com um atributo dominante. O Optuna
convergiu para `entropy`, `max_depth = 6` e `max_features = None`, mantendo todos os atributos
disponíveis. No Random Forest a divergência é ainda maior: Random Search sugeriu uma floresta
**profunda** (`max_depth = 15`, 434 árvores) e o Optuna uma floresta **rasa** (`max_depth = 3`,
380 árvores), que é a escolha mais sensata para esta base — e foi a que teve melhor F1.

Em eficiência, o custo extra do Optuna vem de duas fontes: o TPE é **sequencial** (cada sugestão
depende das anteriores, então não paraleliza a busca como o `n_jobs=-1` do `RandomizedSearchCV`
faz entre combinações) e ainda precisa reajustar o modelo probabilístico a cada trial. Isso
aparece no tempo por avaliação: 0,048 s contra 0,035 s na árvore e 1,94 s contra 1,44 s no Random
Forest. Mesmo assim, 30 s a mais para ganhar meio ponto de F1 numa busca que roda uma única vez é
um preço baixo. **Para este problema o Optuna teve a melhor relação qualidade/tempo**; para
espaços muito maiores ou com orçamento de tempo apertado e muitos núcleos disponíveis, o Random
Search continua competitivo.

### 4.d — Comparação dos métodos de imputação

| Imputação | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| KNN Imputer | 0,8250 | 0,7704 | **0,7750** | **0,7720** | 0,8691 |
| MissForest | **0,8251** | **0,7784** | 0,7583 | 0,7680 | **0,8726** |

**Não houve diferença relevante no desempenho final**: a acurácia é idêntica até a terceira casa
(0,8250 × 0,8251) e a diferença de F1 é de 0,4 p.p., dentro do ruído amostral de um teste com 393
exemplos. O que muda é o perfil: o KNN entrega mais *recall* e o MissForest mais precisão.

Quanto à **preservação da distribuição**, o **KNN Imputer foi melhor**. Em `age`, único atributo
com ausência relevante, ele reduziu o desvio-padrão em apenas 2,78% contra 6,00% do MissForest, e
a distância de Wasserstein foi menor (0,73 × 0,88). Os valores gerados pelo MissForest ficaram
muito concentrados (desvio de 9,4 contra 14,4 do original): por ser uma predição condicional, ele
devolve praticamente a média condicional e achata a cauda da distribuição. O KNN, por copiar
valores reais de vizinhos, preserva melhor a variabilidade. Daí a escolha do **KNN Imputer** como
método adotado — uma escolha justificada pela distribuição, já que o desempenho não desempata.

### 4.e — Comparação das técnicas de balanceamento

| Balanceamento | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| SMOTE | **0,8257** | **0,7805** | 0,7566 | 0,7683 | 0,8708 |
| RandomUnderSampling | 0,8244 | 0,7683 | **0,7767** | **0,7717** | 0,8709 |

O comportamento é exatamente o esperado pela teoria. O **RandomUnderSampling produziu o melhor
equilíbrio para a classe minoritária**, com o maior F1 (0,7717) e o maior recall (0,7767): ao
jogar fora exemplos da classe majoritária, ele desloca a fronteira de decisão em favor dos
sobreviventes e recupera mais deles. O **SMOTE** preserva mais precisão (0,7805) e acurácia
global, porque não descarta informação — ele adiciona exemplos sintéticos por interpolação.

A diferença é pequena (0,34 p.p. de F1) e explica-se pelo desbalanceamento ser moderado (1,62:1).
Fosse uma razão de 10:1, o descarte de dados do *undersampling* provavelmente cobraria um preço
alto. Aqui a base é pequena (916 exemplos de treino) e o RandomUnder descarta 216 deles, ainda
assim sem prejuízo — sinal de que o sinal preditivo está concentrado em poucos atributos e é
redundante entre os exemplos. **Se o objetivo for identificar sobreviventes (recall), a escolha é
o RandomUnderSampling; se for confiar nas previsões positivas (precisão), é o SMOTE.**

### 4.f — As regras continuam interpretáveis no Random Forest?

**Não.** A árvore interpretável produz **8 regras** legíveis em uma página. A árvore otimizada já
sobe para **29 regras**. O Random Forest adotado tem **380 árvores** que somam **milhares de
folhas** — cada uma uma regra, e a previsão final é o voto da maioria de todas elas. Não existe
"a regra" do Random Forest: existe uma média de milhares de regras parcialmente contraditórias,
cada uma treinada num *bootstrap* diferente e vendo um subconjunto diferente de atributos.

O que sobra do RF é a **importância dos atributos**, que é uma explicação *global* (`sex` responde
por 65,5% da importância) mas não *local*: ela não permite responder "por que **este** passageiro
foi classificado como sobrevivente?" nem ser transcrita como uma regra SE-ENTÃO auditável.

A razão de o RF prever melhor é a mesma que o torna opaco. Uma árvore única tem **variância alta**:
mudar alguns exemplos de treino muda os pontos de corte e a estrutura inteira. O RF ataca isso
com *bagging* (cada árvore vê um *bootstrap*) e amostragem de atributos, gerando árvores com erros
pouco correlacionados; na média, a variância cai e o viés quase não sobe. Mas essa média é
justamente o que dissolve a estrutura de regras: o ganho preditivo **vem** da multiplicidade, e a
interpretabilidade **exige** unicidade. É um trade-off estrutural, não um defeito de implementação.

---

## Conclusão

O padrão de sobrevivência do Titanic encontrado pela árvore de decisão otimizada repete e refina o
resultado da Lista 02. O **sexo é o fator dominante** — é a raiz da árvore e responde por 45,5% da
importância na árvore e 65,5% no Random Forest. A regra de maior cobertura do modelo interpretável
diz que **homem com mais de 9,5 anos, de 2ª ou 3ª classe, morreu**: ela sozinha cobre 42% da base
de treino balanceada com 79,9% de confiança. Do lado oposto, **mulher de 1ª ou 2ª classe
sobreviveu**, com 92,2% de confiança entre as que embarcaram em Southampton e **100%** entre as que
embarcaram em Cherbourg ou Queenstown. O protocolo "mulheres e crianças primeiro" aparece
literalmente codificado: a árvore descobriu sozinha, a partir dos dados, o limiar de **9,5 anos**
que separa "criança" de "adulto" no lado masculino — meninos até essa idade e com no máximo 1
irmão a bordo sobreviveram em 86,4% dos casos, enquanto os com famílias maiores morreram em 85,7%.
A classe social modula tudo: mulheres da 3ª classe caem de quase-certeza para 72,1% de
sobrevivência, e as que pagaram tarifa acima de 23,35 nessa mesma classe morreram em 87,5% dos
casos.

Quanto às escolhas de pré-processamento, o **KNN Imputer** foi adotado por preservar melhor a
dispersão de `age` (perda de 2,78% do desvio-padrão contra 6,00% do MissForest), ainda que o
impacto no desempenho final tenha sido nulo — as duas imputações produziram acurácias idênticas até
a terceira casa decimal. Entre as técnicas de balanceamento, o **RandomUnderSampling** entregou o
melhor F1 e recall para a classe minoritária e o **SMOTE** a melhor precisão, com diferença de
apenas 0,34 p.p. entre eles, coerente com o desbalanceamento moderado da base (1,62:1). Na
otimização, a **busca bayesiana (Optuna/TPE) superou o Random Search nos dois modelos** com o mesmo
orçamento de 60 avaliações, encontrando hiperparâmetros qualitativamente diferentes — em especial
uma floresta rasa (`max_depth = 3`) onde o Random Search apostava numa floresta profunda
(`max_depth = 15`).

Por fim, o **Random Forest generaliza melhor** (acurácia 0,8340 contra 0,8162; AUC 0,8854 contra
0,8563; e *gap* treino−teste negativo, contra positivo da árvore), mas ao custo de trocar 8 regras
legíveis por milhares de regras votando em conjunto. O detalhe mais interessante do experimento é
que a **árvore interpretável de profundidade 3 empata com o Random Forest** (acurácia 0,8346, F1
0,7766): num problema em que o sinal está concentrado em três atributos — sexo, classe e idade —
o modelo complexo não tem o que acrescentar. Se o objetivo é prever, o Random Forest é a escolha
segura; se o objetivo é **explicar por que 1.500 pessoas morreram**, a árvore rasa entrega o mesmo
desempenho e ainda pode ser lida em voz alta.
