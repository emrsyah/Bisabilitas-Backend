import puppeteer from 'puppeteer';
import { AxePuppeteer } from '@axe-core/puppeteer';
import { axeRules } from '../config/axeConfig';


export async function analyzeWebsite(url: string) {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 60000 });

  const axeResults = await new AxePuppeteer(page)
    .withRules(axeRules)
    .analyze();

  const content = await page.content();
  await browser.close();

  return { content, axeResults };
}