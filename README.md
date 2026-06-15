# 𝕋𝕣𝕒𝕓𝕒𝕝𝕙𝕠 𝕕𝕖 𝕀𝔸 | 𝗗𝗶𝘀𝗰𝗶𝗽𝗹𝗶𝗻𝗮 𝗱𝗲 𝗜𝗻𝘁𝗲𝗹𝗶𝗴𝗲̂𝗻𝗰𝗶𝗮 𝗔𝗿𝘁𝗶𝗳𝗶𝗰𝗶𝗮𝗹 , 𝗣𝗿𝗼𝗳𝗲𝘀𝘀𝗼𝗿 𝗠𝘂𝗻𝗶𝗳 , 𝗨𝗻𝗶𝗰𝗲𝘀𝘂𝗺𝗮𝗿 𝟮𝟬𝟮𝟲
## Classificação de Spam com KNN e SVM
---
## 𝐈𝐧𝐭𝐞𝐠𝐫𝐚𝐧𝐭𝐞𝐬
- Antonio Ferreira de lima - 23038173-2
- Gabrielle Morente Perna - 23217729-2
- Karen Tanaka - 23026540-2
- Leonardo Leitão Souza- 23020085-2
---
## 𝐂𝐨𝐧𝐭𝐞𝐱𝐭𝐮𝐚𝐥𝐢𝐳𝐚𝐜̧𝐚̃𝐨
Este trabalho tem como foco a aplicação prática de dois algoritmos clássicos de aprendizado de máquina — KNN (K-Nearest Neighbors) e SVM (Support Vector Machine) — para a tarefa de classificação de mensagens SMS como spam ou ham (não-spam).

A escolha dos modelos se deu pois para o tipo de dataset que queríamos, o KNN estudado no 1º bimestre e o SVM Hiperplano estudado no 2º se encaixaram melhor pois ambos funcionam bem com classificação binária — que é exatamente o nosso caso, onde cada mensagem precisa ser classificada em uma de duas categorias. O KNN foi escolhido pela sua simplicidade: ele basicamente olha para as mensagens mais parecidas com a que chegou e decide com base nelas, o que faz bastante sentido para texto. O SVM entrou como segundo modelo porque queríamos comparar essa abordagem mais simples com algo mais robusto — o SVM aprende a traçar uma fronteira entre spam e ham no espaço vetorial e é reconhecido por funcionar muito bem com dados de texto em alta dimensionalidade. A ideia foi colocar os dois para rodar com os mesmos dados e ver na prática qual performa melhor.

E para que os modelos conseguissem trabalhar com texto, as mensagens foram convertidas em vetores numéricos utilizando a técnica TF-IDF (Term Frequency-Inverse Document Frequency), que atribui pesos às palavras com base em sua frequência na mensagem e em sua raridade no conjunto de dados. Palavras muito comuns em todo o dataset recebem peso menor, enquanto palavras mais específicas e relevantes recebem peso maior. Foram considerados unigramas e bigramas (sequências de uma ou duas palavras), com no máximo 5.000 features.

---
## 𝐏𝐫𝐨𝐛𝐥𝐞𝐦𝐚
O problema abordado neste trabalho é a classificação binária de mensagens SMS, com o objetivo de determinar automaticamente se uma mensagem é spam ou ham (não-spam). A classificação manual desse tipo de conteúdo é inviável em larga escala, tornando necessária uma solução automatizada capaz de aprender a partir de exemplos rotulados.

---
## 𝐇𝐢𝐩𝐨́𝐭𝐞𝐬𝐞
Nossa hipótese é que mensagens de spam possuem características linguísticas distintas das mensagens legítimas, como o uso frequente de palavras ligadas a prêmios, urgência, dinheiro e links, além de apresentarem maior comprimento e maior proporção de letras maiúsculas. Com base nessas diferenças, acreditamos que modelos de aprendizado de máquina treinados com representações TF-IDF serão capazes de classificar corretamente a grande maioria das mensagens. Além disso, esperamos que o SVM supere o KNN nessa tarefa, por ser um modelo mais adequado para espaços de alta dimensionalidade como os gerados pela vetorização de texto.

---
## Descrição do Dataset
Foi utilizado o SMS Spam Collection Dataset, disponibilizado publicamente pelo UCI Machine Learning Repository e também disponível no Kaggle.

## Dataset

| Coluna | Tipo | Descrição | Exemplo |
|---|---|---|---|
| `label` | Texto | Classe da mensagem: `ham` (legítima) ou `spam` | `spam` |
| `message` | Texto | Conteúdo completo da mensagem SMS | `"Congratulations! You've won a FREE prize..."` |

Como obter o dataset:
  1. Acesse: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset;
  2. Faça download do arquivo 'spam.csv';
  3. Renomeie para 'SMSSpamCollection.csv' e coloque na mesma pasta deste script;
  OU
  1. Acesse o documento intitulado spam.csv.

---
## 𝐌𝐞́𝐭𝐨𝐝𝐨𝐬 𝐝𝐞 𝐈𝐀 𝐔𝐭𝐢𝐥𝐢𝐳𝐚𝐝𝐨𝐬

### KNN (K-Nearest Neighbors)
O KNN é um algoritmo de aprendizado supervisionado baseado em similaridade. Para classificar uma nova mensagem, ele encontra as k mensagens mais próximas no espaço vetorial e atribui a classe mais comum entre elas.
Configuração utilizada:

k = 5 vizinhos
Métrica de distância: cosseno (mais adequada para vetores TF-IDF esparsos)
Processamento paralelo com n_jobs=-1

O KNN é um modelo simples e interpretável, porém pode ter desempenho limitado em espaços de alta dimensionalidade, como os gerados pela vetorização de texto.

### SVM (Support Vector Machine)
O SVM é um algoritmo de aprendizado supervisionado que busca encontrar o hiperplano de máxima margem que separa as classes no espaço vetorial. A variante utilizada foi o LinearSVC, que é otimizada para problemas de classificação de texto com alta dimensionalidade.
Configuração utilizada:

Kernel linear (LinearSVC)
Parâmetro de regularização: C = 1.0
Máximo de iterações: 2.000

O SVM é especialmente eficiente para problemas de texto, pois funciona bem em espaços com muitas dimensões e pequena quantidade relativa de amostras.

### Pipeline utilizado
Ambos os modelos foram encapsulados em um Pipeline do scikit-learn, garantindo que a vetorização TF-IDF e a classificação ocorram de forma integrada e sem vazamento de dados entre treino e teste.
TF-IDF Vectorizer → [KNN ou SVM]

### 📊 𝐀𝐯𝐚𝐥𝐢𝐚𝐜̧𝐚̃𝐨 𝐝𝐨𝐬 𝐦𝐨𝐝𝐞𝐥𝐨𝐬

| Modelo | Acurácia | Precisão | Revocação | F1-Score |
|---------|---------:|---------:|----------:|---------:|
| KNN | 0.9704 | 0.9531 | 0.8188 | 0.8809 |
| SVM | 0.9830 | 0.9851 | 0.8859 | 0.9329 |

![Análise Exploratória do Dataset](Análise%20Exploratória%20de%20dataset.png)

### 📈 𝐂𝐨𝐦𝐚𝐩𝐫𝐚𝐜̧𝐚̃𝐨 𝐝𝐨𝐬 𝐫𝐞𝐬𝐮𝐥𝐭𝐚𝐝𝐨𝐬

Matriz de confusão

![Matrizes de Confusão](Matrizes%20de%20Confusão%20-%20KNN%20vs%20SVM.png)

Comparação de desempenh0

![Comparação de Desempenho](Comparação%20de%20desempenho%20-%20KNN%20vs%20SVM.png)

---
## 𝐂𝐨𝐧𝐜𝐥𝐮𝐬𝐚̃𝐨
Este trabalho tem como foco a aplicação prática de dois algoritmos clássicos de aprendizado de máquina — KNN (K-Nearest Neighbors) e SVM (Support Vector Machine) — para a tarefa de classificação de mensagens SMS como spam ou ham (não-spam).

Depois de treinar e comparar os dois modelos, ficou claro que o SVM se saiu melhor que o KNN para esse tipo de problema. Isso já era esperado — o SVM é um algoritmo feito para trabalhar bem em espaços com muitas dimensões, que é exatamente o que acontece quando a gente transforma texto em vetores TF-IDF com 5.000 features.

O KNN não foi ruim, mas ele tem uma limitação natural: quanto mais dimensões, mais difícil fica pra ele medir similaridade de forma confiável. Então o desempenho mais fraco faz sentido.
