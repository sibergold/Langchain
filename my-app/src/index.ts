import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import path from 'path'
import {promises as fs} from "fs"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import {PromptTemplate} from "@langchain/core/prompts";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";
import {Ollama} from "@langchain/community/llms/ollama";
import {createRetrievalChain} from "langchain/chains/retrieval";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
const app = new Hono()

app.get('/', (c) => {
  return c.text('Hello Hono!')
})
const ollama = new Ollama({
  baseUrl: "http://localhost:11434", // Default value
  model: "gemma2:2b", // Default value
});
const embeddings = new OllamaEmbeddings({
  model: "gemma2:2b", // default value
  baseUrl: "http://localhost:11434", // default value
  requestOptions: {
    useMMap: true, // use_mmap 1
    numThread: 6, // num_thread 6
    numGpu: 1, // num_gpu 1
  },
});
const getTextFile= async() => {
// const filePath = path.join(__dirname,"../data/langchain-test.txt");
 const filePath = path.join(__dirname,"../data/wsj.txt");
//const filePath = path.join(__dirname,"../data/trendspro-report.pdf");



const data = await fs.readFile(filePath,"utf-8");
return data; 
}

const loadPdfFile=async()=>
{
  const filePath=path.join(__dirname,"../data/trendspro-report.pdf");
  const loader = new PDFLoader(filePath);
  return await loader.load();
}
app.get('/loadPdfEmbedings', async (c) => {
  // Metin dosyasının okunması
  const documents = await loadPdfFile();

  // Vektör veritabanının oluşturulması
  vectorStore = await MemoryVectorStore.fromDocuments(documents, embeddings);

  // Başarı mesajının döndürülmesi
  const response = {message: "Text embeddings loaded successfully."};
  return c.json(response);
})

//Vector Db
let vectorStore:MemoryVectorStore;
app.get('/loadTextEmbedings', async(c)=>
{
  const text = await getTextFile();
  const splitter=new RecursiveCharacterTextSplitter(
    {
      chunkSize:1000,
      separators:['\n\n', '\n', ' ', '', '###'],
      chunkOverlap:50
    });
    const output=await splitter.createDocuments([text])
   
    vectorStore=await MemoryVectorStore.fromDocuments(output,embeddings);
    const response= {message:"Text embedigns is successfuly"};
    return c.json(response);  
  
} )
app.post('/ask', async(c) => {
  const {question} = await c.req.json();
  if(!vectorStore)
  {
    return c.json({message:"Text embedings not loaded yet."});
  }
  const   prompt= PromptTemplate.fromTemplate(
    `You are a helpful AI assistant. Answer the following question based only on the provided context. If the answer cannot be derived from the context, say "I don't have enough information to answer that question." If I like your results I'll tip you $1000!

Context: {context}

Question: {question}

Answer: `
  );


const documentsChain= await createStuffDocumentsChain({
  llm:ollama,
  prompt,
});

const retrievalChain=await createRetrievalChain({
  combineDocsChain:documentsChain,
  retriever:vectorStore.asRetriever({
    k:3
  })
});
const response= await retrievalChain.invoke({
  question:question,
  input:""
});
return c.json({answer:response.answer});

});
const port = 3002
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
