using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace Visma.Hackathon.SecurityAutomation.Web.ML.Models
{
    public partial class InjectionDetectionModel
    {
        public const string RetrainFilePath =  @".\..\TrainingData\SQLInjectionDetection\Modified_SQL_Dataset.csv";
        public const char RetrainSeparatorChar = ',';
        public const bool RetrainHasHeader =  true;

        public static void Train(string outputModelPath, string inputDataFilePath = RetrainFilePath, char separatorChar = RetrainSeparatorChar, bool hasHeader = RetrainHasHeader)
        {
            var mlContext = new MLContext();

            var data = LoadIDataViewFromFile(mlContext, inputDataFilePath, separatorChar, hasHeader);
            var model = RetrainModel(mlContext, data);
            SaveModel(mlContext, model, data, outputModelPath);
        }

        public static IDataView LoadIDataViewFromFile(MLContext mlContext, string inputDataFilePath, char separatorChar, bool hasHeader)
        {
            return mlContext.Data.LoadFromTextFile<InjectionDetectionModel.ModelInput>(inputDataFilePath, separatorChar, hasHeader);
        }
        
        public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
        {
            DataViewSchema dataViewSchema = data.Schema;

            using (var fs = File.Create(modelSavePath))
            {
                mlContext.Model.Save(model, dataViewSchema, fs);
            }
        }

        public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
        {
            var pipeline = BuildPipeline(mlContext);
            var model = pipeline.Fit(trainData);

            return model;
        }

        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName:@"Query",outputColumnName:@"Query")      
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"Query"}))      
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName:@"Label",inputColumnName:@"Label",addKeyValueAnnotationsAsText:false))      
                                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator:mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options(){NumberOfLeaves=473,MinimumExampleCountPerLeaf=2,NumberOfTrees=77,MaximumBinCountPerFeature=39,FeatureFraction=0.242104163857099,LearningRate=0.999999776672986,LabelColumnName=@"Label",FeatureColumnName=@"Features"}),labelColumnName: @"Label"))      
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName:@"PredictedLabel",inputColumnName:@"PredictedLabel"));

            return pipeline;
        }
    }
 }
