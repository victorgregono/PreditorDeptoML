using Microsoft.ML;
using Microsoft.ML.Data;

/// <summary>
/// Espaço de nomes para classificação de departamento
/// </summary>
namespace ClassificadorDepartamento
{
    /// <summary>
    /// Classe para representar os dados da pergunta
    /// </summary>
    public class DadosPergunta
    {
        /// <summary>
        /// Texto da pergunta
        /// </summary>
        [LoadColumn(0)]
        public string TextoPergunta { get; set; }

        /// <summary>
        /// Departamento associado à pergunta
        /// </summary>
        [LoadColumn(1), ColumnName("Label")]
        public string Departamento { get; set; }
    }

    /// <summary>
    /// Classe para representar a previsão do departamento
    /// </summary>
    public class PrevisaoDepartamento
    {
        /// <summary>
        /// Departamento previsto para a pergunta
        /// </summary>
        [ColumnName("PredictedLabel")]
        public string DepartamentoPrevisto { get; set; }
    }

    /// <summary>
    /// Classe principal do programa
    /// </summary>
    class Programa
    {
        /// <summary>
        /// Ponto de entrada principal para o aplicativo.
        /// Este aplicativo é um classificador de departamento.
        /// Ele usa Machine Learning para prever a qual departamento uma determinada pergunta deve ser direcionada.
        /// </summary>
        static void Main()
        {
            var contextoML = new MLContext();

            string diretorioProjeto = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            string caminhoDados = Path.Combine(diretorioProjeto, "MLTraining", "questions.csv");

            var dados = contextoML.Data.LoadFromTextFile<DadosPergunta>(caminhoDados, separatorChar: ';');

            var pipeline = contextoML.Transforms.Text.FeaturizeText("Features", "TextoPergunta")
                .Append(contextoML.Transforms.Conversion.MapValueToKey("Label"))
                .Append(contextoML.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(contextoML.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var modelo = pipeline.Fit(dados);

            var motorPrevisao = contextoML.Model.CreatePredictionEngine<DadosPergunta, PrevisaoDepartamento>(modelo);

            Console.WriteLine("Digite sua pergunta ou digite 'SAIR' para sair.");

            while (true)
            {
                Console.Write("Pergunta: ");
                var entradaUsuario = Console.ReadLine();

                if (entradaUsuario.Equals("SAIR", StringComparison.CurrentCultureIgnoreCase))
                    break;

                var exemploPergunta = new DadosPergunta { TextoPergunta = entradaUsuario };

                var resultado = motorPrevisao.Predict(exemploPergunta);

                Console.WriteLine($"A pergunta: '{exemploPergunta.TextoPergunta}' deve ser direcionada para: {resultado.DepartamentoPrevisto}");
            }
        }
    }
}
