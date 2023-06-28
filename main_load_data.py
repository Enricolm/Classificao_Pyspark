from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
spark = SparkSession.builder.master('local[*]').appName('lendo_modelo').getOrCreate()



dados = spark.read.csv('/home/enricolm/Documents/Pyspark_classificacao/base de dados/dados_clientes.csv',
                    sep = ',',
                    inferSchema = True,
                    header = True
                       )
from pyspark.ml.classification import RandomForestClassificationModel

# Caminho onde o modelo foi salvo
caminho_modelo = '/home/enricolm/Documents/Pyspark_classificacao/modelo'

# Carregar o modelo
modelo_carregado = RandomForestClassificationModel.load(caminho_modelo)

print('-'*40)
print('')


novo_cliente = [{
    'Mais65anos': 0,
    'MesesDeContrato': 1,
    'MesesCobrados': 45.30540797610398,
    'Conjuge': 0,
    'Dependentes': 0,
    'TelefoneFixo': 0,
    'MaisDeUmaLinhaTelefonica': 0,
    'SegurancaOnline': 0,
    'BackupOnline': 0,
    'SeguroDispositivo': 0,
    'SuporteTecnico': 0,
    'TVaCabo': 1,
    'StreamingFilmes': 1,
    'ContaCorreio': 1,
    'Internet_DSL': 1,
    'Internet_FibraOptica': 0,
    'Internet_Nao': 0,
    'TipoContrato_Mensalmente': 1,
    'TipoContrato_UmAno': 0,
    'TipoContrato_DoisAnos': 0,
    'MetodoPagamento_DebitoEmConta': 0,
    'MetodoPagamento_CartaoCredito': 0,
    'MetodoPagamento_BoletoEletronico': 1,
    'MetodoPagamento_Boleto': 0
}]
x=['Mais65anos', 'MesesDeContrato', 'MesesCobrados', 'Conjuge', 'Dependentes', 'TelefoneFixo', 'MaisDeUmaLinhaTelefonica', 'SegurancaOnline', 'BackupOnline', 'SeguroDispositivo', 'SuporteTecnico', 'TVaCabo', 'StreamingFilmes', 'ContaCorreio', 'Internet_DSL', 'Internet_FibraOptica', 'Internet_Nao', 'TipoContrato_Mensalmente', 'TipoContrato_UmAno', 'TipoContrato_DoisAnos', 'MetodoPagamento_DebitoEmConta', 'MetodoPagamento_CartaoCredito', 'MetodoPagamento_BoletoEletronico', 'MetodoPagamento_Boleto']
assembler = VectorAssembler(inputCols=x, outputCol='features')

# %%
novo_cliente_dataset = spark.createDataFrame(novo_cliente)


assembler= assembler.transform(novo_cliente_dataset).select('features')


# %%
valor_final = modelo_carregado.transform(assembler).select('prediction')

valor_final = valor_final.withColumn('prediction',f.when(f.col('prediction') == 1,'Sim' ).otherwise('Nao'))

valor_final.show()
# %%
