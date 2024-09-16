import {
  PuppeteerWebBaseLoader,
} from '@langchain/community/document_loaders/web/puppeteer';
import { Document } from 'langchain/document';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';
import cheerio from 'cheerio';
import { DocumentInterface } from '@langchain/core/documents';

export const scrapeAndCleanData = async (
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
  
export const splitDocuments = async (
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
  