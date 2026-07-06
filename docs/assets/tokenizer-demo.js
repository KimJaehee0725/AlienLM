import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const ORIGINAL_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct";
const ALIEN_TOKENIZER = "dsba-lab/qwen25-7b-instruct-alienlm-full";
const EXAMPLE_TEXT =
  "All happy families are alike; each unhappy family is unhappy in its own way.";

env.allowLocalModels = false;
env.allowRemoteModels = true;

const elements = {
  input: document.querySelector("#demo-input"),
  status: document.querySelector("#demo-status"),
  plainToAlien: document.querySelector("#plain-to-alien"),
  alienToPlain: document.querySelector("#alien-to-plain"),
  roundTrip: document.querySelector("#round-trip"),
  loadExample: document.querySelector("#load-example"),
  translatedText: document.querySelector("#translated-text"),
  recoveredText: document.querySelector("#recovered-text"),
  roundtripCheck: document.querySelector("#roundtrip-check"),
  translatorIds: document.querySelector("#translator-ids"),
  alienNaturalIds: document.querySelector("#alien-natural-ids"),
};

let originalTokenizerPromise;
let alienTokenizerPromise;

function setStatus(message) {
  elements.status.textContent = message;
}

function setBusy(isBusy) {
  for (const button of [
    elements.plainToAlien,
    elements.alienToPlain,
    elements.roundTrip,
    elements.loadExample,
  ]) {
    button.disabled = isBusy;
  }
}

function formatIds(ids) {
  return `[${Array.from(ids).join(", ")}]`;
}

function normalizeText(text) {
  return text.replace(/\r\n/g, "\n");
}

function decode(tokenizer, ids) {
  return tokenizer.decode(ids, {
    skip_special_tokens: false,
    clean_up_tokenization_spaces: false,
  });
}

function encode(tokenizer, text) {
  return tokenizer.encode(text, { add_special_tokens: false });
}

async function getTokenizers() {
  if (!originalTokenizerPromise) {
    originalTokenizerPromise = AutoTokenizer.from_pretrained(ORIGINAL_TOKENIZER);
  }
  if (!alienTokenizerPromise) {
    alienTokenizerPromise = AutoTokenizer.from_pretrained(ALIEN_TOKENIZER);
  }
  return Promise.all([originalTokenizerPromise, alienTokenizerPromise]);
}

function renderResult({
  translated,
  recovered,
  sourceIds,
  alienNaturalIds,
  exact,
  modeLabel,
}) {
  elements.translatedText.textContent = translated || "-";
  elements.recoveredText.textContent = recovered || "-";
  elements.roundtripCheck.textContent = exact
    ? `${modeLabel} round trip: exact match`
    : `${modeLabel} round trip: differs after tokenizer decode/encode`;
  elements.translatorIds.textContent = formatIds(sourceIds);
  elements.alienNaturalIds.textContent = formatIds(alienNaturalIds);
}

async function runPlainToAlien() {
  setBusy(true);
  setStatus("Loading Qwen tokenizers...");
  try {
    const [originalTokenizer, alienTokenizer] = await getTokenizers();
    const plain = normalizeText(elements.input.value);
    const sourceIds = encode(originalTokenizer, plain);
    const translated = decode(alienTokenizer, sourceIds);
    const reencodedIds = encode(alienTokenizer, translated);
    const recovered = decode(originalTokenizer, reencodedIds);
    const alienNaturalIds = encode(alienTokenizer, plain);

    renderResult({
      translated,
      recovered,
      sourceIds,
      alienNaturalIds,
      exact: recovered === plain,
      modeLabel: "Plain to alien",
    });
    setStatus("Plain text translated with the Qwen AlienLM tokenizer.");
  } catch (error) {
    console.error(error);
    setStatus(`Tokenizer demo failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function runAlienToPlain() {
  setBusy(true);
  setStatus("Loading Qwen tokenizers...");
  try {
    const [originalTokenizer, alienTokenizer] = await getTokenizers();
    const alien = normalizeText(elements.input.value);
    const sourceIds = encode(alienTokenizer, alien);
    const translated = decode(originalTokenizer, sourceIds);
    const reencodedIds = encode(originalTokenizer, translated);
    const recovered = decode(alienTokenizer, reencodedIds);
    const alienNaturalIds = encode(alienTokenizer, translated);

    renderResult({
      translated,
      recovered,
      sourceIds,
      alienNaturalIds,
      exact: recovered === alien,
      modeLabel: "Alien to plain",
    });
    setStatus("Alien text recovered with the original Qwen tokenizer.");
  } catch (error) {
    console.error(error);
    setStatus(`Tokenizer demo failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

async function runRoundTrip() {
  setBusy(true);
  setStatus("Loading Qwen tokenizers...");
  try {
    const [originalTokenizer, alienTokenizer] = await getTokenizers();
    const plain = normalizeText(elements.input.value);
    const sourceIds = encode(originalTokenizer, plain);
    const alienText = decode(alienTokenizer, sourceIds);
    const alienIds = encode(alienTokenizer, alienText);
    const recovered = decode(originalTokenizer, alienIds);
    const alienNaturalIds = encode(alienTokenizer, plain);

    renderResult({
      translated: alienText,
      recovered,
      sourceIds,
      alienNaturalIds,
      exact: recovered === plain,
      modeLabel: "Plain to alien to plain",
    });
    setStatus("Round trip completed entirely in the browser.");
  } catch (error) {
    console.error(error);
    setStatus(`Tokenizer demo failed: ${error.message}`);
  } finally {
    setBusy(false);
  }
}

elements.plainToAlien.addEventListener("click", runPlainToAlien);
elements.alienToPlain.addEventListener("click", runAlienToPlain);
elements.roundTrip.addEventListener("click", runRoundTrip);
elements.loadExample.addEventListener("click", () => {
  elements.input.value = EXAMPLE_TEXT;
  setStatus("Example reset. Tokenizers load on first use.");
});
