import express from 'express';
import {
  PuppeteerWebBaseLoader,
  Page,
  Browser,
} from '@langchain/community/document_loaders/web/puppeteer';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { Document } from 'langchain/document';
import {
  JsonOutputParser,
  StringOutputParser,
} from '@langchain/core/output_parsers';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';
import * as cheerio from 'cheerio';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
} from '@langchain/core/prompts';
import { DocumentInterface } from '@langchain/core/documents';
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import {
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables';
import { formatDocumentsAsString } from 'langchain/util/document';
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from '@langchain/google-genai';
import { z } from 'zod';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { CohereRerank } from '@langchain/cohere';
import { downloadMedia } from '../utils/mediaDownloader';
import fs from 'fs';
import {
  transcribeAudio,
  transcribeNonStaticAudio,
} from '../services/transcribeAudio';
import AxePuppeteer from '@axe-core/puppeteer';
// import axe from 'ax'
import puppeteer from 'puppeteer';
import { generateAltText } from '../services/generateAltText';
import { generateAIImprovement } from '../services/generateAIImprovement';


const router = express.Router();

interface SemanticElement {
  type: string;
  content: string;
  children: SemanticElement[];
  attributes: Record<string, string>;
}

const scrapeAndCleanData2 = async (
  url: string,
): Promise<Document<Record<string, any>>[]> => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: 'new',
    },
    async evaluate(page: Page, browser: Browser): Promise<string> {
      const result = await page.evaluate(() => {
        const sections = Array.from(
          document.body.querySelectorAll('h1, h2, h3, h4, h5, h6, p'),
        );
        return JSON.stringify(
          sections.map((section) => ({
            tagName: section.tagName.toLowerCase(),
            textContent: section.textContent,
            id: section.id,
            className: section.className,
          })),
        );
      });
      await browser.close();
      return result;
    },
  });

  const rawData = await loader.load();

  let documents: Document<Record<string, any>>[] = [];
  let currentSection = '';
  let currentContent = '';

  const parsedContent = JSON.parse(rawData[0].pageContent);

  parsedContent.forEach((item: any, index: number) => {
    if (item.tagName.startsWith('h')) {
      if (currentContent) {
        documents.push(
          new Document({
            pageContent: currentContent,
            metadata: {
              url: url,
              section: currentSection,
              sectionId: parsedContent[index - 1]?.id || '',
              sectionClass: parsedContent[index - 1]?.className || '',
            },
          }),
        );
      }
      currentSection = item.textContent;
      currentContent = '';
    } else {
      currentContent += item.textContent + ' ';
    }
  });

  // Add the last section
  if (currentContent) {
    documents.push(
      new Document({
        pageContent: currentContent,
        metadata: {
          url: url,
          section: currentSection,
          sectionId: parsedContent[parsedContent.length - 1]?.id || '',
          sectionClass:
            parsedContent[parsedContent.length - 1]?.className || '',
        },
      }),
    );
  }

  return documents;
};

const scrapeAndCleanDataWStructure = async (
  url: string,
): Promise<Document<Record<string, any>>[]> => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: 'new',
    },
    async evaluate(page: Page, browser: Browser) {
      const htmlContent = await page.content();
      await browser.close();
      return htmlContent;
    },
  });

  const docs = await loader.load();
  const htmlContent = docs[0].pageContent;

  const $ = cheerio.load(htmlContent);

  // Remove script and style tags
  $('script, style').remove();

  const extractSemantics = (element: cheerio.Element): SemanticElement => {
    const tag = element.name;
    const attributes: Record<string, string> = {};

    if ('attribs' in element) {
      Object.entries(element.attribs).forEach(([key, value]) => {
        if (key === 'href' || key === 'src' || key === 'id' || key === 'role') {
          attributes[key] = value;
        }
      });
    }

    const children: SemanticElement[] = [];
    let content = '';

    $(element)
      .contents()
      .each((_, el) => {
        if (el.type === 'text') {
          const text = $(el).text().trim();
          if (text) {
            content += text + ' ';
          }
        } else if (el.type === 'tag') {
          children.push(extractSemantics(el));
        }
      });

    // Determine the semantic type based on the tag and attributes
    let type = tag;
    if (tag === 'nav' || attributes.role === 'navigation') {
      type = 'navigation';
    } else if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag)) {
      type = 'heading';
    } else if (tag === 'a') {
      type = 'link';
    } else if (tag === 'main') {
      type = 'main_content';
    } else if (tag === 'article' || tag === 'section') {
      type = 'content_section';
    } else if (tag === 'aside') {
      type = 'sidebar';
    } else if (tag === 'footer') {
      type = 'footer';
    }

    return { type, content: content.trim(), children, attributes };
  };

  const root = extractSemantics($('body')[0]);

  const semanticToString = (
    node: SemanticElement,
    depth: number = 0,
  ): string => {
    const indent = '  '.repeat(depth);
    let result = `${indent}${node.type}`;

    if (Object.keys(node.attributes).length > 0) {
      const attrs = Object.entries(node.attributes)
        .map(([key, value]) => `${key}="${value}"`)
        .join(' ');
      result += ` (${attrs})`;
    }

    if (node.content) {
      result += `: ${node.content.substring(0, 100)}${
        node.content.length > 100 ? '...' : ''
      }`;
    }

    result += '\n';

    for (const child of node.children) {
      result += semanticToString(child, depth + 1);
    }

    return result;
  };

  const semanticStructure = semanticToString(root);

  const document = new Document({
    pageContent: semanticStructure,
    metadata: {
      url: url,
      contentType: 'text/plain',
    },
  });

  return [document];
};

const scrapeAndCleanData = async (
  url: string,
): Promise<Document<Record<string, any>>[]> => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: 'new',
    },
    // gotoOptions: {
    //   waitUntil: 'networkidle0',
    // },
    async evaluate(page, browser) {
      // Set a random user agent
      // await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');

      // Scroll to bottom to trigger lazy loading
      await page.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight);
      });

      // Wait for 5 seconds to allow dynamic content to load
      // await page.waitForTimeout(5000);

      const content = await page.content();
      await browser.close();
      return content;
    },
  });

  const docs = await loader.load();
  const pageContent = docs[0].pageContent;

  const $ = cheerio.load(pageContent);

  // Remove scripts, styles, and other non-content elements
  $('script, style, nav, footer, header').remove();

  const cleanedText = $('body').html() ?? '';
  const cleaned$ = cheerio.load(cleanedText);

  const textContent = cleaned$('body').text();

  // Remove non-printable characters
  const docsC = textContent.replace(/[^\x20-\x7E]+/g, ' ').trim();

  const docsCNoEmptyLines = docsC.replace(/^\s*[\r\n]/gm, '');

  const documents = [
    new Document({ pageContent: docsCNoEmptyLines, metadata: { url: url } }),
  ];
  return documents;
};

const scrapeForAnalyzer = async (url: string) => {
  let screenshot;
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: { headless: 'new' },
    async evaluate(page) {
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      screenshot = await page.screenshot({
        fullPage: true,
        path: 'screenshot.png',
      });
      const content = await page.content();
      return content;
    },
  });

  const documents = await loader.load();
  const pageContent = documents[0].pageContent;
  const $ = cheerio.load(pageContent);

  $('script, style').remove(); // Remove elements that are not needed

  const buttons = $(
    'button, input[type="button"], input[type="submit"], .button, .btn',
  )
    .map((i, el) => ({
      text: $(el).text().trim(),
      classes: $(el).attr('class')?.split(/\s+/).slice(0, 3).join(' '), // Collect up to the first three class names
      role: $(el).attr('role'),
    }))
    .get();

  const links = $('a')
    .map((i, el) => ({
      href: $(el).attr('href'),
      text: $(el).text().trim() || $(el).attr('aria-label') || '',
      classes: $(el).attr('class')?.split(/\s+/).slice(0, 3).join(' '),
    }))
    .get();

  const inputs = $('input, select, textarea')
    .map((i, el) => ({
      type: $(el).attr('type') || $(el).prop('nodeName').toLowerCase(),
      placeholder: $(el).attr('placeholder'),
      ariaLabel: $(el).attr('aria-label'),
      classes: $(el).attr('class')?.split(/\s+/).slice(0, 3).join(' '),
    }))
    .get();

  const headings = $('h1, h2, h3, h4, h5, h6')
    .map((i, el) => ({
      text: $(el).text().trim(),
      level: $(el).prop('nodeName'),
    }))
    .get();

  const roles = $('[role]')
    .map((i, el) => ({
      role: $(el).attr('role'),
      description: $(el).text().trim() || $(el).attr('aria-label'),
    }))
    .get();

  return {
    buttons,
    links,
    inputs,
    headings,
    roles,
    // screenshot,
  };
};

const splitDocuments = async (
  documents: Document<Record<string, any>>[],
): Promise<DocumentInterface<Record<string, any>>[]> => {
  const transformer = new HtmlToTextTransformer();
  // const transformer2 = new MozillaReadabilityTransformer();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  // const transformer = new HtmlToTextTransformer();
  const sequence = splitter.pipe(transformer);
  const split = await sequence.invoke(documents);
  // console.log('split', split);
  return split;
};

export interface TypedRequestBody<T> extends Express.Request {
  body: T;
}

interface ChatMessage {
  role: 'user' | 'ai';
  content: string;
  evidence: string[];
}

interface AiRequestBody {
  url: string;
  question: string;
  history: ChatMessage[];
}

interface Answer {
  answer: string;
  evidence: string[];
}

type AiResponse = {
  answer: Answer;
  history: ChatMessage[];
};

const getOrCreateVectorStore = async (
  url: string,
  pineconeIndex: Index<RecordMetadata>,
  embeddings:
  | MistralAIEmbeddings
  | GoogleGenerativeAIEmbeddings
  | OpenAIEmbeddings,
): Promise<PineconeStore> => {
  let vectorDimension: number;
  if (process.env.AI_PROVIDER == 'GEMINI') {
    vectorDimension = 768;
  } else if (process.env.AI_PROVIDER == 'MISTRAL') {
    vectorDimension = 1024;
  } else {
    vectorDimension = 1536;
  }
  const zeroVector = new Array(vectorDimension).fill(0);

  const existRecords = await pineconeIndex.query({
    vector: zeroVector,
    topK: 1,
    filter: { url: { $eq: url } },
    includeMetadata: true,
  });

  // console.log(existRecords.matches);

  let vectorStore: PineconeStore;

  if (existRecords.matches.length === 0) {
    // console.log('Creating new vector store for URL:', url);
    const documents = await scrapeAndCleanData(url);
    const splitted = await splitDocuments(documents);
    console.log('Documents:', documents);
    console.log('Splitted documents:', splitted);

    // Create a new vector store and wait for it to be populated
    vectorStore = await PineconeStore.fromDocuments(splitted, embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
      textKey: 'text',
    });

    // Ensure the newly created vectors are immediately available
    // await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second to ensure indexing is complete
  } else {
    // console.log('Using existing vector store for URL:', url);
    vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
      // textKey: 'text',
    });
  }

  // Verify that vectors are present
  // const testQuery = await vectorStore.similaritySearch('test query', 1);
  // console.log('Test query result:', testQuery);

  return vectorStore;
};

interface AIComponents {
  llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI;
  pineconeIndex: any; // Adjust the type according to Pinecone's Index type
  embeddings:
  | MistralAIEmbeddings
  | GoogleGenerativeAIEmbeddings
  | OpenAIEmbeddings;
  reranker: CohereRerank;
}

const aiResponseSchema = z.object({
  answer: z.string().describe('the main response/answer'),
  evidence: z
    .array(z.string())
    .describe('parts of the text from the context used to ground the answer'),
});

const initializeAIComponents = (): AIComponents => {
  const aiProvider = process.env.AI_PROVIDER;

  const pinecone = new Pinecone();
  let llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI;
  let pineconeIndex: any; // Adjust the type according to Pinecone's Index type
  let embeddings:
  | MistralAIEmbeddings
  | GoogleGenerativeAIEmbeddings
  | OpenAIEmbeddings;

  const reranker = new CohereRerank({
    model: 'rerank-multilingual-v3.0',
    topN: 5,
  });

  // console.log(Google)

  if (aiProvider === 'GEMINI') {
    embeddings = new GoogleGenerativeAIEmbeddings();
    llm = new ChatGoogleGenerativeAI({
      model: 'gemini-1.5-flash',
      temperature: 0.4,
    });
    pineconeIndex = pinecone.Index('bisabilitas-768');
  } else if (aiProvider === 'MISTRAL') {
    llm = new ChatMistralAI({
      model: 'mistral-large-latest',
      temperature: 0.4,
    });
    embeddings = new MistralAIEmbeddings();
    pineconeIndex = pinecone.Index('bisabilitas-1024');
  } else {
    llm = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0.4,
    });
    embeddings = new OpenAIEmbeddings();
    pineconeIndex = pinecone.Index('bisabilitas-1536');
  }
  // llm = llm.withStructuredOutput();
  return { llm, pineconeIndex, embeddings, reranker };
};

router.post<{}, AiResponse>(
  '/',
  async (req: TypedRequestBody<AiRequestBody>, res) => {
    const { llm, embeddings, pineconeIndex, reranker } =
      initializeAIComponents();

    // const structuredLLM = llm.withStructuredOutput(aiResponseSchema);

    const { url, history, question } = req.body;

    const vectorStore = await getOrCreateVectorStore(
      url,
      pineconeIndex,
      embeddings,
    );

    const retriever = vectorStore.asRetriever();

    // Contextualize Question Chain
    const contextualizeQSystemPrompt = `Given a chat history and the latest user question
  which might reference context in the chat history, formulate a standalone question
  which can be understood without the chat history. Do NOT answer the question,
  just reformulate it if needed and otherwise return it as is.`;

    const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
      ['system', contextualizeQSystemPrompt],
      new MessagesPlaceholder('chat_history'),
      ['human', '{question}'],
    ]);

    const contextualizeQChain = contextualizeQPrompt
      .pipe(llm)
      .pipe(new StringOutputParser());

    const structuredOutputParser = new JsonOutputParser<
    z.infer<typeof aiResponseSchema>
    >();

    // RAG TEMPLATE
    const template = `
  You are a chrome extension assistant, your role is to help answering people question.
  Use the provided context to answer the question below. Dont fabricate the response. 
  Keep your response concise, with a maximum of four sentences. Respond in JSON format which would be further explained at the end of this prompt. The maximum evidence you provide is 3 evidence
  
  Context: {context}
  
  Question: {question}
  
  Respond ONLY with a JSON object in the format below:
  answer: "string"  // The main answer to the question, derived from the context
  evidence: ["string1", "string2", ...]  // An array of strings containing the context references used for the answer, the maximum word for each evidence should only be 4 words only and it must a copy-paste text from the context.

  JSON Response:
  `;

    const customRagPrompt = PromptTemplate.fromTemplate(template);

    // console.log('retriever', await retriever.invoke(question));

    const ragChain = RunnableSequence.from([
      RunnablePassthrough.assign({
        context: async (input: Record<string, unknown>) => {
          // if (input.chat_history && (input.chat_history as any[]).length > 0) {
          //   return contextualizeQChain.pipe(retriever).pipe(formatDocumentsAsString).invoke(input);
          // }
          // return retriever.invoke(input.question as string).then(formatDocumentsAsString);

          // ================== START ==================
          let docs;
          if (input.chat_history && (input.chat_history as any[]).length > 0) {
            const contextualizedQ = await contextualizeQChain.invoke(input);
            docs = await retriever.invoke(contextualizedQ);
          } else {
            docs = await retriever.invoke(input.question as string);
          }

          console.log(docs);

          // Apply Cohere Rerank
          const rerankedDocs = await reranker.rerank(
            docs,
            input.question as string,
            {
              topN: 5,
            },
          );

          // console.log(rerankedDocs);

          // Sort the reranked documents by relevance score
          const sortedDocs = rerankedDocs
            .sort((a, b) => b.relevanceScore - a.relevanceScore)
            .map((item) => docs[item.index]);

          // console.log(sortedDocs);

          return formatDocumentsAsString(sortedDocs);

          // ================== ENDED ==================
        },
      }),
      customRagPrompt,
      llm,
      structuredOutputParser,
    ]);

    const transformedHistory = history.map((h) => {
      if (h.role == 'ai') {
        return new AIMessage(h.content);
      } else {
        return new HumanMessage(h.content);
      }
    });

    const result = await ragChain.invoke({
      question,
      chat_history: transformedHistory,
    });

    const updatedHistory: ChatMessage[] = [
      ...history,
      { role: 'user', content: question, evidence: [] },
      { role: 'ai', content: result.answer, evidence: result.evidence },
    ];

    res.json({
      answer: {
        answer: result.answer,
        evidence: result.evidence,
      },
      history: updatedHistory,
    });
  },
);

router.post<{}, {}>(
  '/test',
  async (req: TypedRequestBody<AiRequestBody>, res) => {
    const { url } = req.body;
    const docs = await scrapeAndCleanData(url);
    console.log(docs);
    console.log(docs.length);
    res.json({ docs: docs[0].pageContent });
  },
);

router.post<{}, {}>(
  '/analyze',
  async (req: TypedRequestBody<AiRequestBody>, res) => {
    const { url } = req.body;
    const docs = await scrapeForAnalyzer(url);

    const accessibilityTemplate = `
You are an AI trained to enhance web accessibility. Given the content of a webpage divided into sections like headers, buttons, links, and inputs, your task is to evaluate and suggest improvements for web navigation and screen reader compatibility.

Context:
- Headers: {headersJson}
- Buttons: {buttonsJson}
- Links: {linksJson}
- Inputs: {inputsJson}

Question: What improvements can be made to enhance accessibility for keyboard navigation and screen reader users for the given webpage elements?

Respond ONLY with a JSON object in the format below:
- navigationEnhancements: [{
    element: "string", // e.g., "header", "button"
    current: "string", // current state or description
    recommendation: "string" // what should be changed or enhanced
  }]
- screenReaderEnhancements: [{
    element: "string",
    current: "string",
    recommendation: "string"
  }]

JSON Response:
`;

    const adjustedRagPrompt = PromptTemplate.fromTemplate(
      accessibilityTemplate,
    );

    console.log(docs);

    res.json({ docs: docs });
  },
);

router.post<{}, {}>('/transcribe', async (req, res) => {
  let localFilePath: string | null = null;
  try {
    const { mediaUrl } = req.body;

    if (!mediaUrl) {
      return res.status(400).json({ error: 'Media URL is required' });
    }

    localFilePath = await downloadMedia(mediaUrl);
    const transcript = await transcribeAudio(localFilePath);

    res.json({ transcript });
  } catch (error) {
    console.error('Transcription error:', error);
    res.status(500).json({ error: 'An error occurred during transcription' });
  } finally {
    // Clean up the temporary file
    if (localFilePath && fs.existsSync(localFilePath)) {
      fs.unlinkSync(localFilePath);
    }
  }
});

router.post<{}, {}>('/transcribe-audio', async (req, res) => {
  try {
    const bodyData = JSON.parse(req.body);
    const base64Audio = bodyData.audioData;

    const audioBuffer = Buffer.from(base64Audio, 'base64');

    // Use OpenAI API to transcribe the audio
    const transcription = transcribeNonStaticAudio(audioBuffer);

    // Send the transcription text as response
    res.json({ transcription: transcription });
  } catch (error) {
    console.error('Error during transcription:', error);
    res.status(500).send('Error during transcription');
  }
});


router.post<{}, {}>('/improve-accessibility', async (req, res) => {
  try {
    const { url } = req.body;

    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }
    // const puppeteer = new Puppeteer();
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });

    // Perform accessibility analysis
    const axeResults = await new AxePuppeteer(page)
      .withRules([
        'color-contrast',
        'image-alt',
        'aria-required-attr',
        'link-name',
        'button-name',
        'label',
        // 'html-has-lang',
      ])
      .analyze();

    // Get the page content
    const content = await page.content();

    await browser.close();

    const $ = cheerio.load(content);

    // Remove scripts, styles, and other non-content elements
    $('script, style, link, meta').remove();

    // Function to clean an element
    function cleanElement(element: cheerio.Element) {
      // Remove all attributes except for essential ones
      const allowedAttributes = ['href', 'src', 'alt', 'title', 'for', 'type'];
      Object.keys(element.attribs).forEach((attr) => {
        if (!allowedAttributes.includes(attr)) {
          $(element).removeAttr(attr);
        }
      });

      // Recursively clean child elements
      $(element)
        .contents()
        .each((_, el) => {
          if (el.type === 'tag') {
            cleanElement(el);
          }
        });
    }

    // Clean the body element and its children
    cleanElement($('body')[0]);

    // Get the cleaned HTML
    const cleanedHtml = $('body').html() || '';

    // Further cleaning: remove empty elements and whitespace
    const finalHtml = cleanedHtml
      .replace(/>\s+</g, '><') // Remove whitespace between tags
      .replace(/<(\w+)><\/\1>/g, '') // Remove empty elements
      .trim();

    const violationsCriticalAndSerious = axeResults.violations.map((v) => {
      const nodesStructured = v.nodes.map((n) => {
        const cleandedNodes = {
          html: n.html,
          target: n.target,
          improvement: '',
        };
        return cleandedNodes;
      });
      const structured = {
        id: v.id,
        description: v.description,
        impact: v.impact,
        help: v.help,
        helpUrl: v.helpUrl,
        element: nodesStructured,
      };
      return structured;
    });

    const imageAltTextViolations = violationsCriticalAndSerious.filter(
      (v) => v.id === 'image-alt',
    );
    const colorContrastViolations = violationsCriticalAndSerious.filter(
      (v) => v.id === 'color-contrast',
    );
    const ariaViolations = violationsCriticalAndSerious.filter(
      (v) => ((v.id == 'link-name') || (v.id == 'button-name')),
    );

    let finalViolations = [];

    if (imageAltTextViolations.length > 0) {
      const newImageAltText = await Promise.all(imageAltTextViolations[0].element.map(async (el) => {
        let altText: string;
        const regex = /<img[^>]+src="([^"]+)"/;
        const match = el.html.match(regex);
        const src = match ? match[1] : '';
        if (src == '') {
          altText = 'could not find image sources';
        }
        altText = await generateAltText(src) ?? '';
        const updatedHtml = el.html.replace(/<img(.*?)>/, (matches, p1) => {
          // Jika sudah ada alt di dalam tag, ganti nilainya
          if (/alt="[^"]*"/.test(p1)) {
            return matches.replace(/alt="[^"]*"/, `alt="${altText}"`);
          } else {
            // Jika tidak ada, tambahkan alt di akhir
            return `<img${p1} alt="${altText}">`;
          }
        });
        return {
          html: el.html,
          target: el.target,
          improvement: updatedHtml,
        };
      }));
      // console.log(newImageAltText);
      finalViolations.push(
        {
          ...imageAltTextViolations[0],
          element: newImageAltText,
        },
      );
    }

    if (ariaViolations.length > 0) {
      const newAriaViariaViolations = await Promise.all(ariaViolations.map( async (v) => {
        const newDetails = await Promise.all(v.element.map(async (el) => {
          const improvement = await generateAIImprovement(el.html, `Add ONLY ARIA to fix this: ${v.help}, and dont change anything else.`);
          return {
            html: el.html,
            target: el.target,
            improvement: improvement,
          };
        }));
        return {
          ...v,
          element: newDetails,
        };
      }));
      finalViolations.push(
        {
          ...ariaViolations[0],
          element: newAriaViariaViolations,
        },
      );
    }

    finalViolations.push({ ...colorContrastViolations[0] });

    console.log(finalViolations);

    res.json({
      // cleanedHtml: finalHtml,
      accessibilityResults: {
        finalViolations,
      },
    });
  } catch (error) {
    console.error('Error during HTML processing:', error);
    res.status(500).json({ error: 'Error during HTML processing' });
  }
});

router.post<{}, {}>('/check-img', async (req, res) => {
  // const { url } = req.body;
  await generateAltText('https://media.go2speed.org/brand/files/niagahosterid/1/ads-persona-offline-to-online-business-cloud-hosting-affiliate-336-x-280 (1) (1).png');
});

const scrapeAndCleanData3 = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: 'new',
    },
    async evaluate(page, browser) {
      // Scroll to bottom to trigger lazy loading
      await page.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight);
      });

      const content = await page.content();
      await browser.close();
      return content;
    },
  });

  const docs = await loader.load();
  const pageContent = docs[0].pageContent;

  const $ = cheerio.load(pageContent);

  // Remove non-content elements that do not contribute to navigation and accessibility
  $('script, style').remove();

  // Extract relevant elements
  const relevantElements = $('body')
    .find(
      'a, button, input, img, [role], h1, h2, h3, h4, h5, h6, li, ul, ol, section, article',
    )
    .map((i, element) => {
      const el = $(element);
      return {
        tagName: el.prop('tagName'),
        textContent: el.text().trim(),
        htmlContent: el.html()?.trim(),
        attributes: el.attr(),
        parentTag: el.parent().prop('tagName'),
        parentAttributes: el.parent().attr(),
      };
    })
    .get();

  // Prepare document with extracted data
  const documents = [
    new Document({
      pageContent: JSON.stringify(relevantElements),
      metadata: { url: url },
    }),
  ];

  return documents;
};

const scrapeForAccessibilityAI = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: { headless: 'new' },
    async evaluate(page) {
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      const content = await page.content();
      return content;
    },
  });

  const documents = await loader.load();
  const pageContent = documents[0].pageContent;
  const $ = cheerio.load(pageContent);

  $('script, style').remove(); // Remove elements that are not needed

  const structuredData = {
    title: $('title').text(),
    headings: $('h1, h2, h3, h4, h5, h6')
      .map((i, el) => ({
        level: el.name,
        text: $(el).text().trim(),
        index: i,
      }))
      .get(),
    paragraphs: $('p')
      .map((i, el) => ({
        text: $(el).text().trim(),
        index: i,
      }))
      .get(),
    links: $('a')
      .map((i, el) => ({
        href: $(el).attr('href'),
        text: $(el).text().trim() || $(el).attr('aria-label') || '',
        classes: $(el).attr('class')?.split(/\s+/).join(' '),
        index: i,
      }))
      .get(),
    buttons: $(
      'button, input[type="button"], input[type="submit"], .button, .btn',
    )
      .map((i, el) => ({
        text: $(el).text().trim(),
        classes: $(el).attr('class')?.split(/\s+/).join(' '),
        role: $(el).attr('role'),
        index: i,
      }))
      .get(),
    inputs: $('input, select, textarea')
      .map((i, el) => ({
        type: $(el).attr('type') || el.name.toLowerCase(),
        placeholder: $(el).attr('placeholder'),
        ariaLabel: $(el).attr('aria-label'),
        classes: $(el).attr('class')?.split(/\s+/).join(' '),
        index: i,
      }))
      .get(),
    images: $('img')
      .map((i, el) => ({
        src: $(el).attr('src'),
        alt: $(el).attr('alt'),
        width: $(el).attr('width'),
        height: $(el).attr('height'),
        index: i,
      }))
      .get(),
    ariaRoles: $('[role]')
      .map((i, el) => ({
        role: $(el).attr('role'),
        text: $(el).text().trim(),
        ariaLabel: $(el).attr('aria-label'),
        index: i,
      }))
      .get(),
    landmarks: $('header, nav, main, footer, aside, section[aria-label]')
      .map((i, el) => ({
        type: el.name,
        ariaLabel: $(el).attr('aria-label'),
        index: i,
      }))
      .get(),
  };

  return {
    url,
    structuredData,
    fullHtml: $.html(),
  };
};

router.post<{}, {}>(
  '/analyze2',
  async (req: TypedRequestBody<AiRequestBody>, res) => {
    const { url } = req.body;
    const docs = await scrapeAndCleanData3(url);
    console.log(docs[0].pageContent);
    res.json({ docs: docs[0].pageContent });
  },
);

// Define Document class as needed

export default router;
