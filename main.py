from collections import defaultdict
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========================
#     MODELO N-GRAMAS
# ========================

# Necessário baixar o pacote 'punkt' do NLTK
#nltk.download('punkt_tab')

def leitura(nome):
  arq = open(nome, 'r', encoding="utf-8")
  texto = arq.read()
  arq.close()
  return texto


def limpar(lista):
  lixo='.,:;?!"\'()[]{}\/|#$%^&*-'
  quase_limpo = [x.strip(lixo).lower() for x in lista]
  return [x for x in quase_limpo if x.isalpha() or '-' in x]

def preprocessar_texto(nome):
    texto = leitura(nome)
    limpar(texto)
    sentencas = nltk.sent_tokenize(texto)
    return [nltk.word_tokenize(sent) for sent in sentencas]


corpus_base = preprocessar_texto('corpus_base.txt')
corpus_limpo = [limpar(s) for s in corpus_base]

with open("corpus_preparado.txt", "w", encoding="utf-8") as arq:
    for s in corpus_limpo:
        arq.write(" ".join(s) + "\n") 

with open('corpus_preparado.txt','r', encoding="utf-8") as arq:
  sentencas = arq.readlines()

vocab = set()
contagens = defaultdict(int)
for linha in sentencas:
    sent = linha.split()
    for palavra in sent:
        vocab |= {palavra}
        contagens[palavra] += 1

hapax = [ p for p in contagens.keys() if contagens[p]== 1]
vocab -= set(hapax)
vocab |= {'<DES>'}

unigramas = defaultdict(int)
bigramas = defaultdict(int)
trigramas = defaultdict(int)

def ngramas(n, sent):
    return[tuple(sent[i:i+n]) for i in range(len(sent) - n + 1)]

for linha in sentencas:
    sent = linha.split()
    for i in range(len(sent)):
        if sent[i] in hapax:
            sent[i] = '<DES>'

    uni = ngramas(1,sent)
    bi = ngramas(2,sent)
    tri = ngramas(3, sent)
    for x in uni:
        unigramas[x] += 1
    for x in bi:
        bigramas[x] += 1
    for x in tri:
        trigramas[x] += 1

def prob_uni(x):
    V = len(vocab)
    N = sum(unigramas.values())
    return((unigramas[x] + 1) / (N + V))

def prob_bi(x):
    V = len(vocab)
    return ((bigramas[x] + 1) / (unigramas[(x[0],)] + V))

def prob_tri(x):
    V = len(vocab)
    return ((trigramas[x] + 1) / (bigramas[(x[0],x[1])] + V))

def obter_palavras_unicas(ordem):
    palavras_unicas = []
    for p in ordem:
        if p[1] not in palavras_unicas:
            palavras_unicas.append(p[1])
        if len(palavras_unicas) == 3:
            break
    return palavras_unicas

def prever(palavra):
  palavra = palavra.lower()
  resultado_trigrama = [ ch for ch in trigramas.keys() if ch[0] == palavra ]
  if resultado_trigrama:
    ordem = sorted(resultado_trigrama, key=lambda x:prob_tri(x), reverse=True)
    palavras_unicas = obter_palavras_unicas(ordem)
    return " | ".join(palavras_unicas)
    
  resultado_bigrama = [ ch for ch in bigramas.keys() if ch[0] == palavra ]
  if resultado_bigrama:
    ordem = sorted(resultado_bigrama, key=lambda x:prob_bi(x), reverse=True)
    palavras_unicas = obter_palavras_unicas(ordem)
    return " | ".join(palavras_unicas)
  
  return "Não há previsão"

# ========================
#     MODELO TUCANO
# ========================

model_id = "TucanoBR/Tucano-2b4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

def prever_tucano(palavra):
    try:
        inputs = tokenizer.encode(palavra, return_tensors="pt", truncation=True, max_length=100).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        logits = outputs.logits[:, -1, :]
        top_tokens = torch.topk(logits, 10, dim=-1).indices.squeeze().tolist()
        
        palavras = []
        for token in top_tokens:
            palavra = tokenizer.decode([token], skip_special_tokens=True).strip()
            if palavra:
                palavras.append(palavra)

        if len(palavras) == 0:
            return "Sem previsão"
        return " | ".join(palavras[:3])
    
    except Exception as e:
        return "Erro na predição: " + str(e)

# ========================
# Execução principal
# ========================

print("Selecione o modelo:")
print("1 - N-Gramas")
print("2 - Tucano")
modelo = input("Opção: ")

entrada = input("\nDigite palavras para predição (para finalizar, digite #): ")
while entrada != "#":
    if modelo == "1":
        print(prever(entrada))
    elif modelo == "2":
        print(prever_tucano(entrada))
    else:
        print("Opção inválida!")
    entrada = input()