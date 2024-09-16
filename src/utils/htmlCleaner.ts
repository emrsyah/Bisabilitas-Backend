import * as cheerio from 'cheerio';

const cleanElement = (element: cheerio.Element, $: cheerio.CheerioAPI) => {
  const allowedAttributes = ['href', 'src', 'alt', 'title', 'for', 'type'];
  Object.keys(element.attribs).forEach((attr) => {
    if (!allowedAttributes.includes(attr)) {
      $(element).removeAttr(attr);
    }
  });

  $(element).contents().each((_, el) => {
    if (el.type === 'tag') {
      cleanElement(el, $);
    }
  });
};

export const cleanHtml = (content: string): string => {
  const $ = cheerio.load(content);
  $('script, style, link, meta').remove();
  cleanElement($('body')[0], $);
  return $('body').html() || '';
};