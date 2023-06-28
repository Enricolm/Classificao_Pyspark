from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').appName('criacao_modelo').getOrCreate()



dados = spark.read.csv('/home/enricolm/Documents/Pyspark_classificacao/base de dados/dados_clientes.csv',
                    sep = ',',
                    inferSchema = True,
                    header = True
                       )


import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression,DecisionTreeClassifier,RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


rfc_best = RandomForestClassifier(maxDepth=22,maxBins=36,numTrees=50)





colunasBinarias = [
    'Churn',
    'Conjuge',
    'Dependentes',
    'TelefoneFixo',
    'MaisDeUmaLinhaTelefonica',
    'SegurancaOnline',
    'BackupOnline',
    'SeguroDispositivo',
    'SuporteTecnico',
    'TVaCabo',
    'StreamingFilmes',
    'ContaCorreio'
]


dados_transformados = [f.when(f.col(c) == 'Sim',1).otherwise(0).alias(c)for c in colunasBinarias]


for col in reversed(dados.columns):
    if col not in colunasBinarias:
        dados_transformados.insert(0,col)



dados_transformados = dados.select(dados_transformados)




internet = dados_transformados.groupBy('id').pivot('Internet').agg(f.lit(1)).na.fill(0)

# %%
TipoContrato = dados_transformados.groupBy('id').pivot('TipoContrato').agg(f.lit(1)).na.fill(0)

# %%
MetodoPagamento = dados_transformados.groupBy('id').pivot('MetodoPagamento').agg(f.lit(1)).na.fill(0)

# %%
dataset = dados_transformados\
    .join(internet, 'id' , how = 'inner')\
    .join(MetodoPagamento, 'id', how = 'inner')\
    .join(TipoContrato,'id' , how = 'inner')\
    .select('*',
        f.col('DSL').alias('Internet_DSL'), 
        f.col('FibraOptica').alias('Internet_FibraOptica'), 
        f.col('Nao').alias('Internet_Nao'), 
        f.col('Mensalmente').alias('TipoContrato_Mensalmente'), 
        f.col('UmAno').alias('TipoContrato_UmAno'), 
        f.col('DoisAnos').alias('TipoContrato_DoisAnos'), 
        f.col('DebitoEmConta').alias('MetodoPagamento_DebitoEmConta'), 
        f.col('CartaoCredito').alias('MetodoPagamento_CartaoCredito'), 
        f.col('BoletoEletronico').alias('MetodoPagamento_BoletoEletronico'), 
        f.col('Boleto').alias('MetodoPagamento_Boleto'))\
    .drop(
        'Internet', 'TipoContrato', 'MetodoPagamento', 'DSL', 
        'FibraOptica', 'Nao', 'Mensalmente', 'UmAno', 'DoisAnos', 
        'DebitoEmConta', 'CartaoCredito', 'BoletoEletronico', 'Boleto'
    )


dataset = dataset.withColumnRenamed('Churn', 'label')
x = dataset.columns
x.remove('label')
x.remove('id')


assembler = VectorAssembler(inputCols=x, outputCol='features')

dataset_trans = assembler.transform(dataset).select('features','label')




model_rfc_best = rfc_best.fit(dataset_trans)

print(x)

model_rfc_best.save('/home/enricolm/Documents/Pyspark_classificacao/modelo')



