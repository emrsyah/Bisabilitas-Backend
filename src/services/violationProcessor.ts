import { ensureURL } from '../utils/urlHelper';
import { Violation, ViolationElement } from '../interfaces/Accesibility';
import { generateAIImprovement } from './generateAIImprovement';
import { generateAltText } from './generateAltText';

function categorizeViolations(violations: Violation[]) {
  return {
    imageAlt: violations.filter((v) => v.id === 'image-alt'),
    colorContrast: violations.filter((v) => v.id === 'color-contrast'),
    aria: violations.filter((v) =>
      [
        'link-name',
        'button-name',
        'aria-required-attr',
        'label',
        'input-button-name',
        'select-name',
        'frame-title',
        'aria-valid-attr-value',
      ].includes(v.id),
    ),
    other: violations.filter(
      (v) =>
        ![
          'image-alt',
          'color-contrast',
          'link-name',
          'button-name',
          'aria-required-attr',
          'label',
          'input-button-name',
          'select-name',
          'frame-title',
          'aria-valid-attr-value',
        ].includes(v.id),
    ),
  };
}

async function processImageAltViolations(
  violation: Violation,
  baseUrl: string,
) {
  const newImageAltText = await Promise.all(
    violation.element.map(async (el: ViolationElement) => {
      const regex = /<img[^>]+src="([^"]+)"/;
      const match = el.html.match(regex);
      let src = match ? match[1] : '';
      let altText = 'Image description unavailable';

      if (src) {
        try {
          altText = (await generateAltText(ensureURL(src, baseUrl))) || altText;
        } catch (error) {
          console.error(`Failed to generate alt text for ${src}:`, error);
        }
      } else {
        console.warn('Could not find image source in HTML:', el.html);
      }

      const updatedHtml = el.html.replace(
        /<img(.*?)>/,
        (matches: string, p1: string) => {
          return /alt="[^"]*"/.test(p1)
            ? matches.replace(/alt="[^"]*"/, `alt="${altText}"`)
            : `<img${p1} alt="${altText}">`;
        },
      );

      return { ...el, improvement: updatedHtml };
    }),
  );

  return { ...violation, element: newImageAltText };
}

async function processAriaViolations(violations: Violation[]) {
  return Promise.all(
    violations.map(async (v) => ({
      ...v,
      element: await Promise.all(
        v.element.map(async (el: ViolationElement) => ({
          ...el,
          improvement: await generateAIImprovement(
            el.html,
            `Add ONLY ARIA to fix this: ${v.help}, and don't change anything else.`,
          ),
        })),
      ),
    })),
  );
}

async function processOtherViolations(violations: Violation[]) {
  return Promise.all(
    violations.map(async (v) => ({
      ...v,
      element: await Promise.all(
        v.element.map(async (el) => ({
          ...el,
          improvement: await generateAIImprovement(
            el.html,
            `${v.help}, only add, edit, or remove any attribute inside the HTML tag, don't change anything else.`,
          ),
        })),
      ),
    })),
  );
}

export async function processViolations(
  violations: Violation[],
  baseUrl: string,
) {
  const categorizedViolations = categorizeViolations(violations);
  const finalViolations = [];

  if (categorizedViolations.imageAlt.length > 0) {
    finalViolations.push(
      await processImageAltViolations(
        categorizedViolations.imageAlt[0],
        baseUrl,
      ),
    );
  }

  if (categorizedViolations.aria.length > 0) {
    finalViolations.push(
      ...(await processAriaViolations(categorizedViolations.aria)),
    );
  }

  if (categorizedViolations.other.length > 0) {
    finalViolations.push(
      ...(await processOtherViolations(categorizedViolations.other)),
    );
  }

  finalViolations.push(...categorizedViolations.colorContrast);

  return finalViolations;
}
