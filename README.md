# Classificador de Departamento

Este projeto é um classificador de departamento que usa Machine Learning para prever a qual departamento uma determinada pergunta deve ser direcionada.

## Como funciona

O programa é construído usando o ML.NET, uma biblioteca de Machine Learning para .NET. Ele carrega um conjunto de dados de perguntas e seus respectivos departamentos, treina um modelo de classificação multiclasse e, em seguida, usa esse modelo para prever o departamento para novas perguntas.

## Estrutura do Código

O código principal está no método Main da classe Programa. Aqui está um resumo do que cada parte do código faz:

- `contextoML = new MLContext();`: Cria um novo contexto de Machine Learning. O contexto de ML é usado para todas as operações de ML.NET.
- `diretorioProjeto = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;`: Obtém o diretório do projeto.
- `caminhoDados = Path.Combine(diretorioProjeto, "MLTraining", "questions.csv");`: Constrói o caminho para o arquivo de dados.
- `dados = contextoML.Data.LoadFromTextFile<DadosPergunta>(caminhoDados, separatorChar: ';');`: Carrega os dados do arquivo CSV.
- O bloco de código que começa com `pipeline = contextoML.Transforms.Text.FeaturizeText("Features", "TextoPergunta")...`: Cria um pipeline de treinamento. Este pipeline transforma o texto das perguntas em características numéricas, mapeia os rótulos de string para chaves, treina um classificador multiclasse e mapeia as chaves previstas de volta para strings.
- `modelo = pipeline.Fit(dados);`: Treina o modelo usando os dados carregados.
- `motorPrevisao = contextoML.Model.CreatePredictionEngine<DadosPergunta, PrevisaoDepartamento>(modelo);`: Cria um motor de previsão. O motor de previsão é usado para fazer previsões com o modelo.
- O bloco de código que começa com `while (true)...`: Este é o loop principal do programa. Ele solicita ao usuário que insira uma pergunta, faz uma previsão usando o motor de previsão e exibe o departamento previsto.

## Como usar

Para usar o programa, basta executá-lo, digitar uma pergunta quando solicitado e o programa irá dizer a qual departamento a pergunta deve ser direcionada. Para sair do programa, digite 'SAIR'.
