# PROJETO INTEGRADOR - AEM1



library(tidyverse)
library(rsample)
library(pROC)
library(yardstick)
library(glmnet)
library(ranger)
library(xgboost)


# Carregamento e conferencia da base -----------------------------------------

base <- read_csv(
  "base_final.csv") 


cat("Dimensoes da base:", nrow(base), "linhas e", ncol(base), "colunas\n")

distribuicao_resposta <- base |>
  count(deixou_de_operar) |>
  mutate(
    frequencia_relativa = n / sum(n),
    porcentagem = 100 * frequencia_relativa
  )

print(distribuicao_resposta)


# Divisao em treino e teste --------------------------------------------------

# 80% para ajustar os modelos e 20% para avaliar as previsoes.
set.seed(123)
divisao <- initial_split(
  base,
  prop = 0.80,
  strata = deixou_de_operar
)

treino <- training(divisao)
teste <- testing(divisao)

cat("Treino:", nrow(treino), "observacoes\n")
cat("Teste:", nrow(teste), "observacoes\n")


# Ponto de corte principal ---------------------------------------------------

# Definimos inicialmente utilizar o corte de 0.20 porque queremos aumentar a
# sensibilidade e reduzir falsos negativos.
# Ao final, utilizamos pROC::coords() para mostrar o efeito
# de outros cortes entre 0.10 e 0.50.
corte_principal <- 0.20


# Funcao para avaliar cada modelo ------------------------------------

avaliar_modelo <- function(
  nome_modelo,
  observado,
  probabilidade,
  ponto_corte = 0.20
) {
  previsto <- ifelse(probabilidade >= ponto_corte, 1, 0)

  verdadeiro_positivo <- sum(observado == 1 & previsto == 1)
  falso_negativo <- sum(observado == 1 & previsto == 0)
  verdadeiro_negativo <- sum(observado == 0 & previsto == 0)
  falso_positivo <- sum(observado == 0 & previsto == 1)

  curva_roc <- pROC::roc(
    observado,
    probabilidade,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )

  sensibilidade <- verdadeiro_positivo /
    (verdadeiro_positivo + falso_negativo)

  especificidade <- verdadeiro_negativo /
    (verdadeiro_negativo + falso_positivo)

  if (verdadeiro_positivo + falso_positivo == 0) {
    precisao <- NA_real_
  } else {
    precisao <- verdadeiro_positivo /
      (verdadeiro_positivo + falso_positivo)
  }

  tibble(
    modelo = nome_modelo,
    ponto_corte = ponto_corte,
    auc = as.numeric(pROC::auc(curva_roc)),
    sensibilidade = sensibilidade,
    taxa_falsos_negativos = 1 - sensibilidade,
    especificidade = especificidade,
    precisao = precisao,
    verdadeiro_positivo = verdadeiro_positivo,
    falso_negativo = falso_negativo,
    falso_positivo = falso_positivo,
    verdadeiro_negativo = verdadeiro_negativo
  )
}


# Modelo nulo ---------------------------------------------------------------

# O modelo nulo usa apenas a proporcao da resposta e funciona como referencia.
modelo_nulo <- glm(
  deixou_de_operar ~ 1,
  data = treino,
  family = "binomial"
)

probabilidade_nulo <- predict(
  modelo_nulo,
  newdata = teste,
  type = "response"
) |>
  as.numeric()


# Regressoes logisticas ------------------------------------------------------

# Sao ajustadas duas especificacoes para nao inserir simultaneamente liquidez
# e capital de giro, que carregam informacoes contabeis relacionadas.

formula_logistica_sem_liquidez <- deixou_de_operar ~ . - liquidez

formula_logistica_sem_capital <- deixou_de_operar ~ . - log_capital_de_giro

modelo_logistico_sem_liquidez <- glm(
  formula_logistica_sem_liquidez,
  data = treino,
  family = "binomial"
)

modelo_logistico_sem_capital <- glm(
  formula_logistica_sem_capital,
  data = treino,
  family = "binomial"
)



probabilidade_logistica_sem_liquidez <- predict(
  modelo_logistico_sem_liquidez,
  newdata = teste,
  type = "response"
) |>
  as.numeric()

probabilidade_logistica_sem_capital <- predict(
  modelo_logistico_sem_capital,
  newdata = teste,
  type = "response"
) |>
  as.numeric()


# Matrizes para Ridge, LASSO e XGBoost --------------------------------------


x_treino <- model.matrix(
  deixou_de_operar ~ .,
  data = treino
)[, -1]

x_teste <- model.matrix(
  deixou_de_operar ~ .,
  data = teste
)[, -1]

y_treino <- treino$deixou_de_operar

if (!identical(colnames(x_treino), colnames(x_teste))) {
  stop("As matrizes de treino e teste possuem colunas diferentes.")
}


# Ridge ----------------------------------------------------------------------

set.seed(123)
cv_ridge <- cv.glmnet(
  x = x_treino,
  y = y_treino,
  family = "binomial",
  alpha = 0,
  nfolds = 10,
  type.measure = "auc",
  standardize = TRUE
)

probabilidade_ridge <- predict(
  cv_ridge,
  newx = x_teste,
  s = "lambda.1se",
  type = "response"
) |>
  as.numeric()


# LASSO ----------------------------------------------------------------------


set.seed(123)
cv_lasso <- cv.glmnet(
  x = x_treino,
  y = y_treino,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  type.measure = "auc",
  standardize = TRUE
)

probabilidade_lasso <- predict(
  cv_lasso,
  newx = x_teste,
  s = "lambda.1se",
  type = "response"
) |>
  as.numeric()

matriz_coeficientes_lasso <- as.matrix(
  coef(cv_lasso, s = "lambda.1se")
)

coeficientes_lasso <- tibble(
  variavel = rownames(matriz_coeficientes_lasso),
  coeficiente = as.numeric(matriz_coeficientes_lasso[, 1])
) |>
  filter(coeficiente != 0) |>
  arrange(desc(abs(coeficiente)))

print(coeficientes_lasso, n = Inf)


# Floresta aleatoria ---------------------------------------------------------


treino_floresta <- treino |>
  mutate(
    deixou_de_operar = factor(
      deixou_de_operar,
      levels = c(0, 1),
      labels = c("permaneceu", "fechou")
    )
  )

teste_floresta <- teste |>
  mutate(
    deixou_de_operar = factor(
      deixou_de_operar,
      levels = c(0, 1),
      labels = c("permaneceu", "fechou")
    )
  )

# O erro OOB permite comparar configuracoes usando somente a base de treino.
# Testamos mtry (4, 6, 8, 10) e numero de arvores (100, 250, 500)
grade_floresta <- crossing(
  mtry = c(4, 6, 8, 10),
  n_arvores = c(100, 250, 500)
) |>
  mutate(erro_oob = NA_real_)

set.seed(123)
for (i in 1:nrow(grade_floresta)) {
  modelo_temporario <- ranger(
    deixou_de_operar ~ .,
    data = treino_floresta,
    num.trees = grade_floresta$n_arvores[i],
    mtry = grade_floresta$mtry[i],
    seed = 123
  )

  grade_floresta$erro_oob[i] <- modelo_temporario$prediction.error
}

melhor_floresta <- grade_floresta |>
  arrange(erro_oob) |>
  slice(1)

print(grade_floresta, n = Inf)
print(melhor_floresta)

# O modelo final usa os melhores parametros e retorna probabilidades.
set.seed(123)
modelo_floresta <- ranger(
  deixou_de_operar ~ .,
  data = treino_floresta,
  probability = TRUE,
  num.trees = melhor_floresta$n_arvores,
  mtry = melhor_floresta$mtry,
  importance = "permutation",
  seed = 123
)

probabilidade_floresta <- predict(
  modelo_floresta,
  data = teste_floresta
)$predictions[, "fechou"] |>
  as.numeric()


# XGBoost --------------------------------------------------------------------


ajustar_xgboost <- function(splits, eta, nrounds, max_depth) {
  dados_ajuste <- training(splits)
  dados_avaliacao <- testing(splits)

  matriz_ajuste <- model.matrix(
    deixou_de_operar ~ .,
    data = dados_ajuste
  )[, -1]

  matriz_avaliacao <- model.matrix(
    deixou_de_operar ~ .,
    data = dados_avaliacao
  )[, -1]

  d_ajuste <- xgb.DMatrix(
    data = matriz_ajuste,
    label = dados_ajuste$deixou_de_operar
  )

  d_avaliacao <- xgb.DMatrix(
    data = matriz_avaliacao,
    label = dados_avaliacao$deixou_de_operar
  )

  
  modelo <- xgb.train(
    data = d_ajuste,
    nrounds = nrounds,
    verbose = 0,
    params = list(
      max_depth = max_depth,
      eta = eta,
      nthread = 3,
      objective = "binary:logistic",
      eval_metric = "auc"
    )
  )

  probabilidade <- predict(modelo, d_avaliacao)

  curva_roc <- pROC::roc(
    dados_avaliacao$deixou_de_operar,
    probabilidade,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )

  as.numeric(pROC::auc(curva_roc))
}


# Grade de hiperparâmetros
grade_xgboost <- crossing(
  eta = c(0.01, 0.10),
  nrounds = c(250, 750),
  max_depth = c(1, 4)
)

set.seed(123)
dobras_xgboost <- vfold_cv(
  treino,
  v = 5,
  strata = deixou_de_operar
)

set.seed(123)
resultados_xgboost <- dobras_xgboost |>
  crossing(grade_xgboost) |>
  mutate(
    auc = pmap_dbl(
      list(splits, eta, nrounds, max_depth),
      ajustar_xgboost
    )
  )

resumo_xgboost <- resultados_xgboost |>
  group_by(eta, nrounds, max_depth) |>
  summarise(
    auc_media = mean(auc),
    .groups = "drop"
  ) |>
  arrange(desc(auc_media))

melhor_xgboost <- resumo_xgboost |>
  slice(1)

print(resumo_xgboost, n = Inf)
print(melhor_xgboost)

d_treino <- xgb.DMatrix(
  data = x_treino,
  label = y_treino
)

d_teste <- xgb.DMatrix(
  data = x_teste,
  label = teste$deixou_de_operar
)

set.seed(123)
modelo_xgboost <- xgb.train(
  data = d_treino,
  nrounds = melhor_xgboost$nrounds,
  verbose = 0,
  params = list(
    max_depth = melhor_xgboost$max_depth,
    eta = melhor_xgboost$eta,
    nthread = 3,
    objective = "binary:logistic",
    eval_metric = "auc"
  )
)

probabilidade_xgboost <- predict(
  modelo_xgboost,
  d_teste
)


# Avaliacao final no teste ---------------------------------------------------


observado <- teste$deixou_de_operar

resultados_modelos <- bind_rows(
  avaliar_modelo(
    "Modelo nulo",
    observado,
    probabilidade_nulo,
    corte_principal
  ),
  avaliar_modelo(
    "Logistica sem liquidez",
    observado,
    probabilidade_logistica_sem_liquidez,
    corte_principal
  ),
  avaliar_modelo(
    "Logistica sem capital de giro",
    observado,
    probabilidade_logistica_sem_capital,
    corte_principal
  ),
  avaliar_modelo(
    "Ridge",
    observado,
    probabilidade_ridge,
    corte_principal
  ),
  avaliar_modelo(
    "LASSO",
    observado,
    probabilidade_lasso,
    corte_principal
  ),
  avaliar_modelo(
    "Floresta aleatoria",
    observado,
    probabilidade_floresta,
    corte_principal
  ),
  avaliar_modelo(
    "XGBoost",
    observado,
    probabilidade_xgboost,
    corte_principal
  )
) |>
  arrange(desc(auc), desc(sensibilidade))

print(resultados_modelos, n = Inf, width = Inf)



# Analise complementar dos pontos de corte com pROC::coords() ----------------


avaliar_cortes <- function(nome_modelo, observado, probabilidade) {
  curva_roc <- pROC::roc(
    observado,
    probabilidade,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )

  pROC::coords(
    curva_roc,
    x = seq(0.10, 0.50, by = 0.05),
    input = "threshold",
    ret = c(
      "threshold",
      "accuracy",
      "sensitivity",
      "specificity",
      "ppv",
      "npv"
    ),
    transpose = FALSE
  ) |>
    as_tibble() |>
    mutate(modelo = nome_modelo, .before = 1)
}

tabela_cortes <- bind_rows(
  avaliar_cortes(
    "Logistica sem liquidez",
    observado,
    probabilidade_logistica_sem_liquidez
  ),
  avaliar_cortes(
    "Logistica sem capital de giro",
    observado,
    probabilidade_logistica_sem_capital
  ),
  avaliar_cortes("Ridge", observado, probabilidade_ridge),
  avaliar_cortes("LASSO", observado, probabilidade_lasso),
  avaliar_cortes(
    "Floresta aleatoria",
    observado,
    probabilidade_floresta
  ),
  avaliar_cortes("XGBoost", observado, probabilidade_xgboost)
)

print(tabela_cortes, n = Inf, width = Inf)




# Graficos de comparacao -----------------------------------------------------

resultados_graficos <- resultados_modelos |>
  filter(modelo != "Modelo nulo")

grafico_auc <- resultados_graficos |>
  ggplot(aes(x = reorder(modelo, auc), y = auc)) +
  geom_col(fill = "#386CB0", width = 0.70) +
  geom_text(
    aes(label = sprintf("%.3f", auc)),
    hjust = -0.15,
    size = 3.8
  ) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Comparacao dos modelos pela AUC",
    subtitle = "Modelos ajustados no treino e avaliados no teste",
    x = "Modelo",
    y = "AUC"
  ) +
  theme_minimal(base_size = 12)

print(grafico_auc)

grafico_sensibilidade <- resultados_graficos |>
  ggplot(
    aes(
      x = reorder(modelo, sensibilidade),
      y = sensibilidade
    )
  ) +
  geom_col(fill = "#33A65C", width = 0.70) +
  geom_text(
    aes(label = sprintf("%.3f", sensibilidade)),
    hjust = -0.15,
    size = 3.8
  ) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Comparacao dos modelos pela sensibilidade",
    subtitle = "Resultado com ponto de corte igual a 0.20",
    x = "Modelo",
    y = "Sensibilidade"
  ) +
  theme_minimal(base_size = 12)

print(grafico_sensibilidade)


# Curvas ROC -----------------------------------------------------------------

predicoes_modelos <- tibble(
  observado = factor(
    observado,
    levels = c(0, 1),
    labels = c("permaneceu", "fechou")
  ),
  `Logistica sem liquidez` = probabilidade_logistica_sem_liquidez,
  `Logistica sem capital de giro` = probabilidade_logistica_sem_capital,
  Ridge = probabilidade_ridge,
  LASSO = probabilidade_lasso,
  `Floresta aleatoria` = probabilidade_floresta,
  XGBoost = probabilidade_xgboost
) |>
  pivot_longer(
    cols = -observado,
    names_to = "modelo",
    values_to = "probabilidade"
  )

grafico_roc <- predicoes_modelos |>
  group_by(modelo) |>
  yardstick::roc_curve(
    observado,
    probabilidade,
    event_level = "second"
  ) |>
  autoplot() +
  labs(
    title = "Curvas ROC dos modelos",
    x = "1 - especificidade",
    y = "Sensibilidade",
    color = "Modelo"
  ) +
  theme_bw()

print(grafico_roc)


# Importancia das preditoras -------------------------------------------------


preparar_importancia <- function(nomes, valores, numero_variaveis = 10) {
  tibble(
    variavel = nomes,
    importancia = as.numeric(valores)
  ) |>
    filter(
      variavel != "(Intercept)",
      is.finite(importancia),
      importancia > 0
    ) |>
    mutate(
      variavel = case_when(
        str_starts(variavel, "faixa_inoffice_days") ~
          "faixa_inoffice_days",

        str_starts(variavel, "geracao_media_ceos") ~
          "geracao_media_ceos",

        str_starts(variavel, "porte_empresa") ~
          "porte_empresa",

        str_starts(variavel, "region_m") ~
          "region_m",

        str_starts(variavel, "gender") ~
          "gender",

        str_starts(variavel, "origin") ~
          "origin",

        str_starts(variavel, "ind2") ~
          "ind2",

        TRUE ~ variavel
      )
    ) |>
    group_by(variavel) |>
    summarise(
      # Soma a contribuição das categorias pertencentes à mesma preditora.
      importancia = sum(importancia),
      .groups = "drop"
    ) |>
    arrange(desc(importancia)) |>
    slice_head(n = numero_variaveis) |>
    mutate(
      importancia = 100 * importancia / max(importancia)
    )
}


grafico_importancia <- function(dados, titulo) {
  dados |>
    arrange(importancia) |>
    mutate(
      variavel = factor(variavel, levels = variavel)
    ) |>
    ggplot(aes(x = variavel, y = importancia)) +
    geom_col(fill = "#7A5195", width = 0.70) +
    coord_flip() +
    labs(
      title = titulo,
      subtitle = "Importancia relativa por preditora dentro do modelo",
      x = "Preditor",
      y = "Importancia relativa - maximo = 100"
    ) +
    theme_minimal(base_size = 11)
}


# Regressoes: usamos |coeficiente| multiplicado pelo desvio-padrao da coluna.
matriz_logistica_sem_liquidez <- model.matrix(
  modelo_logistico_sem_liquidez
)
coeficientes_logistica_sem_liquidez <- coef(
  modelo_logistico_sem_liquidez
)
nomes_logistica_sem_liquidez <- intersect(
  colnames(matriz_logistica_sem_liquidez),
  names(coeficientes_logistica_sem_liquidez)
) |>
  setdiff("(Intercept)")

importancia_logistica_sem_liquidez <- preparar_importancia(
  nomes_logistica_sem_liquidez,
  abs(coeficientes_logistica_sem_liquidez[nomes_logistica_sem_liquidez]) *
    map_dbl(
      as_tibble(
        matriz_logistica_sem_liquidez[
          , nomes_logistica_sem_liquidez,
          drop = FALSE
        ]
      ),
      sd
    )
)

matriz_logistica_sem_capital <- model.matrix(
  modelo_logistico_sem_capital
)
coeficientes_logistica_sem_capital <- coef(
  modelo_logistico_sem_capital
)
nomes_logistica_sem_capital <- intersect(
  colnames(matriz_logistica_sem_capital),
  names(coeficientes_logistica_sem_capital)
) |>
  setdiff("(Intercept)")

importancia_logistica_sem_capital <- preparar_importancia(
  nomes_logistica_sem_capital,
  abs(coeficientes_logistica_sem_capital[nomes_logistica_sem_capital]) *
    map_dbl(
      as_tibble(
        matriz_logistica_sem_capital[
          , nomes_logistica_sem_capital,
          drop = FALSE
        ]
      ),
      sd
    )
)

# Ridge e LASSO: coeficientes calculados em lambda.1se.
coeficientes_ridge <- as.matrix(
  coef(cv_ridge, s = "lambda.1se")
)[, 1]
nomes_ridge <- intersect(
  colnames(x_treino),
  names(coeficientes_ridge)
)

importancia_ridge <- preparar_importancia(
  nomes_ridge,
  abs(coeficientes_ridge[nomes_ridge]) *
    map_dbl(
      as_tibble(x_treino[, nomes_ridge, drop = FALSE]),
      sd
    )
)

coeficientes_lasso_vetor <- as.matrix(
  coef(cv_lasso, s = "lambda.1se")
)[, 1]
nomes_lasso <- intersect(
  colnames(x_treino),
  names(coeficientes_lasso_vetor)
)

importancia_lasso <- preparar_importancia(
  nomes_lasso,
  abs(coeficientes_lasso_vetor[nomes_lasso]) *
    map_dbl(
      as_tibble(x_treino[, nomes_lasso, drop = FALSE]),
      sd
    )
)

# Floresta: importancia por permutacao.
importancia_floresta <- preparar_importancia(
  names(modelo_floresta$variable.importance),
  modelo_floresta$variable.importance
)

# XGBoost: Gain mede a contribuicao das divisoes de cada preditora.
tabela_importancia_xgboost <- xgb.importance(
  feature_names = colnames(x_treino),
  model = modelo_xgboost
)

importancia_xgboost <- preparar_importancia(
  tabela_importancia_xgboost$Feature,
  tabela_importancia_xgboost$Gain
)

print(
  grafico_importancia(
    importancia_logistica_sem_liquidez,
    "Logistica sem liquidez"
  )
)
print(
  grafico_importancia(
    importancia_logistica_sem_capital,
    "Logistica sem capital de giro"
  )
)
print(grafico_importancia(importancia_ridge, "Ridge"))
print(grafico_importancia(importancia_lasso, "LASSO"))
print(grafico_importancia(importancia_floresta, "Floresta aleatoria"))
print(
  grafico_importancia(
    importancia_xgboost,
    "VIP do XGBoost"
  )
)


# Conclusao automatica -------------------------------------------------------

# A Floresta Aleatoria e selecionada porque apresentou a maior sensibilidade
# no corte 0.20. Essa escolha acompanha o objetivo principal do projeto:
# reduzir falsos negativos, mesmo que o XGBoost apresente AUC maior.
modelo_selecionado <- resultados_modelos |>
  filter(modelo == "Floresta aleatoria")

resultado_xgboost <- resultados_modelos |>
  filter(modelo == "XGBoost")

cat("\nCONCLUSAO FINAL\n")
cat(
  "O modelo selecionado foi", modelo_selecionado$modelo,
  "com AUC =", sprintf("%.3f", modelo_selecionado$auc),
  "e sensibilidade =", sprintf("%.3f", modelo_selecionado$sensibilidade),
  "no corte", sprintf("%.2f", modelo_selecionado$ponto_corte), ".\n"
)
cat(
  "Ele identificou", modelo_selecionado$verdadeiro_positivo,
  "das",
  modelo_selecionado$verdadeiro_positivo +
    modelo_selecionado$falso_negativo,
  "empresas que deixaram de operar.\n"
)
cat(
  "O XGBoost apresentou AUC maior =",
  sprintf("%.3f", resultado_xgboost$auc),
  ", mas sensibilidade menor =",
  sprintf("%.3f", resultado_xgboost$sensibilidade), ".\n"
)
cat(
  "Como a prioridade e reduzir falsos negativos, a maior sensibilidade da",
  "Floresta Aleatoria determinou a escolha final.\n"
)
cat(
  "Os resultados medem capacidade preditiva na amostra de teste e nao",
  "representam relacoes causais.\n"
)
