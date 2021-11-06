# Regressão Logística

## Jupyter Demos (Exemplos em _inglês_)

▶️ [Demo | Regressão Logística com Limites Lineares](https://nbviewer.jupyter.org/github/aceiro/homemade-machine-learning/blob/master/notebooks/logistic_regression/logistic_regression_with_linear_boundary_demo_ptBR.ipynb) - predição da Iris `class` baseada em `petal_length` e `petal_width`

▶️ [Demo | Regressão Logística sem Limites Lineares](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/logistic_regression_with_non_linear_boundary_demo.ipynb) - predição microchip `validity` baseada em `param_1` e `param_2`

▶️ [Demo | Regressão Logística Multivariada | MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/multivariate_logistic_regression_demo.ipynb) - reconhecedor de dígitos escritos manualmente em imagens `28x28` pixels.

▶️ [Demo | Regressão Logística Multivariada | Fashion MNIST](https://nbviewer.jupyter.org/github/trekhleb/homemade-machine-learning/blob/master/notebooks/logistic_regression/multivariate_logistic_regression_fashion_demo.ipynb) - reconhecedor de tipos de roupas em imagens `28x28` pixels

## Definição

**Regressão Logística** é uma análise apropriada para ser conduzida quando variáveis dependentes são dicotômicas (i.e., binárias). Como todas as análises de regressão, a regressão logística é uma análise preditiva. Regressão Logística é usada para descrever dados e explicar o relacionamento entre uma variável dependente e uma ou mais variáveis nominais, ordignárias, intervalos ou variáveis níveis de proporção independentes.

A Regressão Logística é usada quando as variáveis dependentes (target) são categoricas.

Por exemplo:
- Para prever se um e-mail é spam (1) ou não (0)
- Para prever se uma peça tem defeito (1) ou não (0)
- Para determinar se um tumor é malígno (1) ou não (0)

Em outras palavras, a saída de uma regressão logística pode ser entendida como:

![Logistic Regression Output](../../images/logistic_regression/output.svg)

![Logistic Regression](https://cdn-images-1.medium.com/max/1600/1*4G0gsu92rPhN-co9pv1P5A@2x.png)

![Logistic Regression](https://cdn-images-1.medium.com/max/1200/1*KRhpHnucyX9Y5PMdjGvVFA.png)

## Conjunto de dados de Treinamento

O conjunto de treinamento é um dado de entrada onde para cada conjunto predefinido de recursos _x_ temos uma classificação correta _y_.

![Training Set](../../images/logistic_regression/training-set-1.svg)

_m_ - número de exemplos no trainamento

![Training Set](../../images/logistic_regression/training-set-2.svg)

Por simplicidade, define-se:

![x-zero](../../images/logistic_regression/x-0.svg)

![Saída da Regressão Logística](../../images/logistic_regression/output.svg)

## Hipótese (o Modelo)

A equação que obtém recursos e parâmetros como uma entrada e prevê o valor como uma saída (ou seja, prever se o e-mail é spam ou não com base em algumas características do e-mail).

![Hipótese](../../images/logistic_regression/hypothesis-1.svg)

Onde _g()_ is a **função sigmóide**.

![Sigmoid](../../images/logistic_regression/sigmoid.svg)

![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

Podemos escrever a hipótese como:

![Hipótese](../../images/logistic_regression/hypothesis-2.svg)

![Predizer 0](../../images/logistic_regression/predict-0.svg)

![Predizer 1](../../images/logistic_regression/predict-1.svg)

## Função Custo

Função que mostra quão precisas são as previsões da hipótese com o conjunto atual de parâmetros.

![Função Custo](../../images/logistic_regression/cost-function-1.svg)

![Função Custo](../../images/logistic_regression/cost-function-4.svg)

![Função Custo](../../images/logistic_regression/cost-function-2.svg)

A Função Custo pode ser simplificado como:

![Função Custo](../../images/logistic_regression/cost-function-3.svg)

## Gradiente Descendente em Lote

O gradiente descendente é um algoritmo de otimização iterativa para encontrar o mínimo de uma função de custo descrita acima. Para encontrar um mínimo local de uma função usando a descida de gradiente, deve-se dar passos proporcionais ao negativo do gradiente (ou gradiente aproximado) da função no ponto atual.

A Figura abaixo ilustra os passos que damos na descida do gradiente para encontrar o mínimo local.

![Gradiente Descendente](https://cdn-images-1.medium.com/max/1600/1*f9a162GhpMbiTVTAua_lLQ.png)

A direção da etapa é definida pela derivada da função de custo no ponto atual.

![Gradiente Descendente](https://cdn-images-1.medium.com/max/1600/0*rBQI7uBhBKE8KT-X.png)

Depois de decidir a direção que precisamos seguir, precisamos decidir qual o tamanho do passo que precisamos dar.

![Gradiente Descendente](https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png)

Precisamos atualizar simultaneamente ![Theta](../../images/logistic_regression/theta-j.svg) for _j = 0, 1, ..., n_

![Gradiente Descendente](../../images/logistic_regression/gradient-descent-1.svg)

![Gradiente Descendente](../../images/logistic_regression/gradient-descent-2.svg)

![alpha](../../images/logistic_regression/alpha.svg) - a taxa de aprendizagem, a constante que define o tamanho do degrau de descida do gradiente

![x-i-j](../../images/logistic_regression/x-i-j.svg) - _j<sup>th</sup>_ característica do valor do _i<sup>th</sup>_ examplo de treinamento

![x-i](../../images/logistic_regression/x-i.svg) - entrada (características) de _i<sup>th</sup>_ exemplo de treinamento

_y<sup>i</sup>_ - saída do _i<sup>th</sup>_ exemplo de treinamento

_m_ - número de exemplos de treinamento

_n_ - número de características

> Quando usamos o termo "lote" para descida gradiente, significa que cada passo de descida gradiente usa **all** os exemplos de treinamento (como você pode ver na fórmula acima).
> 
## Classificação Multi-class (One-vs-All)

Muitas vezes, precisamos fazer não apenas classificações binárias (0/1), mas sim multiclasses, como:

- Tempo: ensolarado, nublado, chuva, neve
- Marcação de e-mail: trabalho, amigos, família

Para lidar com este tipo de problemas, podemos treinar um classificador de regressão logística ![Classificador Multi-class](../../images/logistic_regression/multi-class-classifier.svg) várias vezes para cada classe _i_ para predizer a probabilidade de _y = i_.

![One-vs-All](https://i.stack.imgur.com/zKpJy.jpg)

## Regularização

### Problema de Overfitting (Sobreajuste)

If we have too many features, the learned hypothesis may fit the **training** set very well:

![overfitting](../../images/logistic_regression/overfitting-1.svg)

**But** it may fail to generalize to **new** examples (let's say predict prices on new example of detecting if new messages are spam).

![overfitting](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/fittings.jpg)

### Solução para Overfitting

Aqui estão algumas opções que podem ser abordadas:

- Reduza o número de características
     - Selecione manualmente quais características manter
     - Algoritmo de seleção de modelo
- Regularização
     - Manter todos as características, mas reduzir magnitude / valores dos parâmetros do modelo (thetas).
     - Funciona bem quando temos muitas características, cada um dos quais contribui um pouco para a previsão de _y_.

A regularização funciona adicionando o parâmetro de regularização à ** função de custo **:

![Função Custo](../../images/logistic_regression/cost-function-with-regularization.svg)

![Parametros de Regulação](../../images/logistic_regression/lambda.svg) - Parametros de Regulação

> Observe que você não deve regularizar o parâmetro ![theta zero](../../images/logistic_regression/theta-0.svg).

Nesse caso, a fórmula de ** gradiente descendente ** será semelhante à seguinte:

![Gradiente Descendente](../../images/logistic_regression/gradient-descent-3.svg)

## Referencias (inglês)

- [Machine Learning on Coursera](https://www.coursera.org/learn/machine-learning)
- [Sigmoid Function on Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)
- [Gradient Descent on Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [Gradient Descent by Suryansh S.](https://hackernoon.com/gradient-descent-aynk-7cbe95a778da)
- [Gradient Descent by Niklas Donges](https://towardsdatascience.com/gradient-descent-in-a-nutshell-eaf8c18212f0)
- [One vs All on Stackexchange](https://stats.stackexchange.com/questions/318520/many-binary-classifiers-vs-single-multiclass-classifier)
- [Logistic Regression by Rohan Kapur](https://ayearofai.com/rohan-1-when-would-i-even-use-a-quadratic-equation-in-the-real-world-13f379edab3b)
- [Overfitting on GeeksForGeeks](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/)
