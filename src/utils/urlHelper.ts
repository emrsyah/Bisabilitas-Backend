export function ensureURL(text: string, baserUrl: string): string {
  try {
    // Check if the string is a valid URL
    new URL(text);
    return text;
  } catch {
    // If not a valid URL, add something in front (e.g., "https://example.com")
    return `${baserUrl}${text}`;
  }
}
