const $ = id => document.getElementById(id);
const links = [...document.querySelectorAll('nav a')];
const panels = [...document.querySelectorAll('.panel')];
const upload = $('upload');
const search = $('search');
const searchButton = search.querySelector('button');
let book;
let result;
let browserMode = false;
let browserLibrary;
let uploadLimit = 500 * 1024 * 1024;

function selectStep() {
  const requested = new URLSearchParams(location.search).get('step');
  const step = panels.some(panel => panel.id === requested) ? requested : 'text-splitting';
  for (const panel of panels) panel.hidden = panel.id !== step;
  for (const link of links) {
    if (new URL(link.href).searchParams.get('step') === step) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
}
for (const link of [...links, ...document.querySelectorAll('.continue')]) {
  link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    history.pushState(null, '', link.href);
    selectStep();
  });
}
window.addEventListener('popstate', selectStep);
selectStep();

async function request(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error('Could not connect to the server. Check your connection and retry.');
  }
  if (response.status === 413) {
    throw new Error('This PDF exceeds the upload limit. Choose a smaller PDF or split it into chapters.');
  }
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('json') ? await response.json().catch(() => null) : null;
  if (typeof data?.detail === 'string' && !response.ok) throw new Error(data.detail);
  if (response.status === 504) throw new Error('The server timed out. Try a shorter PDF or retry the request.');
  if (!response.ok) throw new Error('The server could not complete the request (HTTP ' + response.status + ').');
  if (data === null) throw new Error('The server returned an unreadable response. Please retry.');
  return data;
}

request('/capabilities').then(({hosted, large_uploads}) => {
  if (!hosted) return;
  browserMode = true;
  $('upload-limit').textContent = 'Text PDFs up to ' + '500 MB, processed on this device' + ' · Indexes are temporary';
  upload.elements.hybrid.disabled = false;
  $('answer-mode').value = 'extractive';
  const generated = $('answer-mode').querySelector('[value=generated]');
  generated.disabled = true;
  generated.textContent = 'Generated explanation (requires local Nemotron)';
}).catch(() => {});

function element(tag, text, className) {
  const item = document.createElement(tag);
  item.textContent = text;
  if (className) item.className = className;
  return item;
}

function sourceCard(chunk, text, meta, id) {
  const card = element('article', text, 'card');
  if (id) card.id = id;
  const details = element('small', 'Chunk ' + chunk.id + ' · ' + meta + ' · ');
  const link = element('a', 'Open PDF page ' + chunk.page);
  link.href = (book.pdfUrl || '/books/' + book.id + '/pdf') + '#page=' + chunk.page;
  link.target = '_blank';
  link.rel = 'noopener';
  details.append(link);
  card.append(details);
  return card;
}

function drawPlot(container, points, query) {
  container.replaceChildren();
  if (!points.length && !query) return;
  const coords = query ? [...points, query] : points;
  const xs = coords.map(point => point.x);
  const ys = coords.map(point => point.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const width = Math.max(...xs) - minX;
  const height = Math.max(...ys) - minY;
  const position = point => [
    width ? 7 + 86 * (point.x - minX) / width : 50,
    height ? 93 - 86 * (point.y - minY) / height : 50,
  ];
  const namespace = 'http://www.w3.org/2000/svg';
  if (query) {
    const svg = document.createElementNS(namespace, 'svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('aria-hidden', 'true');
    const [x, y] = position(query);
    for (const point of points) {
      const [px, py] = position(point);
      const line = document.createElementNS(namespace, 'line');
      for (const [key, value] of Object.entries({x1:x, y1:y, x2:px, y2:py})) line.setAttribute(key, value);
      line.setAttribute('stroke', '#888');
      line.setAttribute('stroke-dasharray', '1 1');
      line.setAttribute('vector-effect', 'non-scaling-stroke');
      svg.append(line);
    }
    container.append(svg);
  }
  for (const point of coords) {
    const [x, y] = position(point);
    const dot = element('button', '', point === query ? 'question' : '');
    const label = point === query ? 'Question vector' : 'Chunk ' + point.id + ' · PDF page ' + point.page;
    dot.type = 'button';
    dot.title = label;
    dot.setAttribute('aria-label', label);
    dot.style.left = x + '%';
    dot.style.top = y + '%';
    if (point === query) {
      dot.disabled = true;
    } else {
      dot.onclick = () => {
        const card = $(container.id + '-' + point.id);
        card.parentElement.querySelector('.selected')?.classList.remove('selected');
        card.classList.add('selected');
        card.scrollIntoView({block:'nearest'});
      };
    }
    container.append(dot);
  }
}

function resetSearch() {
  result = undefined;
  $('citation-evidence').hidden = true;
  $('answer').classList.remove('ready');
  $('answer').textContent = 'Search a textbook first.';
  $('matches').textContent = 'Run a search to see ranked passages.';
  $('search-status').textContent = '';
  $('context').replaceChildren();
  $('query-vector').hidden = true;
  $('search-plot').replaceChildren();
  $('proof').replaceChildren(element('li', 'Waiting for a textbook and question.'));
  $('generate').disabled = true;
  $('open-answer').hidden = true;
}

upload.addEventListener('submit', async event => {
  event.preventDefault();
  const data = new FormData(upload);
  const file = data.get('file');
  if (file.size > uploadLimit) {
    $('index-status').textContent = '“' + file.name + '” exceeds the '
      + (uploadLimit === 4_000_000 ? '4 MB hosted' : '500 MB')
      + ' upload limit. Choose a smaller PDF or split it into chapters.';
    return;
  }
  if (Number(data.get('overlap')) >= Number(data.get('size'))) {
    $('index-status').textContent = 'Overlap must be smaller than chunk size.';
    return;
  }
  const controls = [...upload.elements].filter(control => !control.disabled);
  controls.forEach(control => control.disabled = true);
  searchButton.disabled = true;
  $('generate').disabled = true;
  $('index-status').textContent = 'Uploading and indexing. Large books may take a few minutes.';
  try {
    let nextBook;
    if (browserMode) {
      browserLibrary = await import('/browser-library.js?v=evidence-ids-v4');
      nextBook = await browserLibrary.index(file, data, (page, total) => {
        $('index-status').textContent = typeof page === 'string' ? page : 'Reading locally · page ' + page + ' / ' + total;
      });
    } else {
      nextBook = await request('/books', {method:'POST', body:data});
    }
    if (book?.pdfUrl) URL.revokeObjectURL(book.pdfUrl);
    showBook(nextBook);
    if (!browserMode) refreshLibrary();
  } catch (error) {
    $('index-status').textContent = error.message + (book ? ' Still using “' + book.name + '”.' : '');
  } finally {
    controls.forEach(control => control.disabled = false);
    searchButton.disabled = !book;
    $('generate').disabled = !result?.matches.length;
  }
});

search.addEventListener('submit', async event => {
  event.preventDefault();
  if (!book) return;
  const data = new FormData(search);
  const question = data.get('question').trim();
  if (question.length < 2) {
    $('search-status').textContent = 'Enter a study question.';
    return;
  }
  resetSearch();
  const locked = [...search.elements, ...upload.elements].filter(control => !control.disabled);
  locked.forEach(control => control.disabled = true);
  $('search-status').textContent = 'Ranking passages…';
  try {
    result = browserMode ? await browserLibrary.search(question, Number(data.get('limit'))) : await request('/books/' + book.id + '/search', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question, limit:Number(data.get('limit'))}),
    });
    result.question = question;
    $('prompt-question').textContent = question;
    const vector = $('query-vector');
    vector.textContent = '[' + result.query_vector.map(value => value.toFixed(4)).join(', ')
      + (result.dimensions > result.query_vector.length ? ', …' : '') + '] · ' + result.dimensions + ' dimensions';
    vector.hidden = false;
    result.limit = Number(data.get('limit'));
    const {matches, point, searched} = result;
    $('search-status').textContent = 'Compared your question with ' + searched + ' chunks from ' + book.name + '.';
    $('proof').replaceChildren(
      element('li', 'Indexed ' + book.textPages + ' textbook pages into ' + searched + ' chunks.'),
      element('li', 'Ranked chunks for “' + question + '”.'),
      element('li', matches.length ? 'Retrieved ' + matches.length + ' passages from PDF pages '
        + [...new Set(matches.map(match => match.page))].join(', ') + '.' : 'No matching passages found.'),
    );
    if (!matches.length) {
      $('matches').textContent = 'No matching passage found. Try a specific term from the textbook.';
      $('answer').textContent = 'There is insufficient textbook evidence to answer this question.';
      return;
    }
    $('matches').replaceChildren(...matches.map(match => sourceCard(match, match.text,
      result.score_type + ' ' + match.score.toFixed(4), 'search-plot-' + match.id)));
    $('context').replaceChildren(...matches.map(match => sourceCard(match, match.excerpt, 'Retrieved context')));
    drawPlot($('search-plot'), matches, {x:point[0], y:point[1]});
    $('answer').textContent = 'Select Build answer to assemble the cited passages.';
    $('generate').disabled = false;
    $('open-answer').hidden = false;
  } catch (error) {
    $('search-status').textContent = error.message;
  } finally {
    locked.forEach(control => control.disabled = false);
  }
});

$('generate').onclick = async () => {
  if (!result?.matches.length) return;
  const controls = [...search.elements, ...upload.elements, $('generate'), $('answer-mode'), $('temperature'), $('max-tokens')]
    .filter(control => !control.disabled);
  if (!['temperature', 'max-tokens'].every(id => $(id).value !== '' && $(id).reportValidity())) return;
  controls.forEach(control => control.disabled = true);
  $('generate').textContent = 'Generating…';
  const output = $('answer');
  const proof = $('proof');
  proof.querySelector('.complete')?.remove();
  $('citation-evidence').hidden = true;
  output.classList.remove('ready');
  output.textContent = 'Checking textbook evidence…';
  try {
    const data = browserMode ? ($('answer-mode').value === 'generated'
      ? await browserLibrary.generate(result, Number($('temperature').value), Number($('max-tokens').value))
      : browserLibrary.answer(result)) : await request('/books/' + book.id + '/answer', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: result.question, limit: result.limit, mode: $('answer-mode').value, temperature: Number($('temperature').value), max_tokens: Number($('max-tokens').value)}),
    });
    if (!data.claims.length) {
      output.textContent = data.limitation || (data.generation_unverified
        ? 'The model could not verify an explanation from these matches. Review the retrieved passages in Semantic Search.'
        : 'There is insufficient textbook evidence to answer this question.');
      return;
    }
    output.classList.add('ready');
    output.replaceChildren();
    if (data.generation_unverified) {
      output.append(element('p', 'The model could not verify an explanation. Showing exact textbook sentences instead.', 'coverage'));
    }
    if (data.mode === 'generated') output.append(element('h3', (data.title || result.question).toUpperCase(), 'answer-title'));
    const panel = $('citation-evidence');
    panel.replaceChildren(element('h3', 'Highlighted textbook evidence'));
    panel.hidden = false;
    let previousHeading = '';
    data.claims.forEach((claim, index) => {
      if (claim.heading && claim.heading !== previousHeading) {
        output.append(element('h4', claim.heading, 'answer-heading'));
        previousHeading = claim.heading;
      }
      const paragraph = element('p', '', 'answer-paragraph answer-' + (['point','flow','takeaway'].includes(claim.kind) ? claim.kind : 'paragraph'));
      if (typeof claim.lead === 'string' && claim.lead.trim()) {
        paragraph.append(element('strong', claim.lead.trim() + ' — '));
      }
      paragraph.append(claim.text);
      const citation = element('button', '[' + (index + 1) + ']', 'citation inline-citation');
      citation.type = 'button';
      citation.setAttribute('aria-label', 'Show citation ' + (index + 1) + ' on PDF page ' + claim.page);
      const evidence = element('article', '', 'card');
      evidence.tabIndex = -1;
      const offset = Array.from(claim.excerpt).slice(0, claim.offset).join('').length;
      const excerpt = element('p', claim.excerpt.slice(0, offset), 'quote');
      excerpt.append(element('mark', claim.quote), claim.excerpt.slice(offset + claim.quote.length));
      const link = element('a', 'Open original PDF page ' + claim.page);
      link.href = (book.pdfUrl || '/books/' + book.id + '/pdf') + '#page=' + claim.page;
      link.target = '_blank';
      link.rel = 'noopener';
      evidence.append(element('h3', '[' + (index + 1) + '] PDF page ' + claim.page), excerpt, link);
      panel.append(evidence);
      citation.onclick = () => evidence.focus();
      paragraph.append(' ', citation);
      output.append(paragraph);
    });
    if (data.mode !== 'generated' && data.unmatched_terms.length) {
      output.append(element('div', 'Question terms absent from the supporting quotes: '
        + data.unmatched_terms.join(', ') + '.', 'coverage'));
    }
    if (data.limitation) output.append(element('p', data.limitation, 'coverage'));
    proof.append(element('li', 'Verified ' + data.claims.length + ' exact textbook quotes'
      + (data.mode === 'generated' && data.support_check !== 'citations-only' ? ' and checked claim support with the local model.' : '.'), 'complete'));
  } catch (error) {
    output.textContent = error.message;
  } finally {
    $('generate').textContent = 'Build answer';
    controls.forEach(control => control.disabled = false);
  }
};

function showBook(nextBook) {
  book = nextBook;

    resetSearch();
    const coverage = book.textPages + ' of ' + book.pages + ' PDF pages contained selectable text.';
    $('index-status').textContent = 'Index ready. ' + coverage;
    $('open-vectors').hidden = false;
    $('book-info').textContent = book.name + ' · ' + coverage;
    const details = element('details', '');
    details.append(element('summary', 'Embedding details'), element('p', book.model.includes('LangChain')
      ? 'Transformers.js runs the quantized all-MiniLM-L6-v2 model in this browser. LangChain ranks normalized passage and query vectors using cosine similarity.'
      : book.model.includes('all-minilm')
      ? 'Ollama encodes chunks with all-minilm and handles tokenization internally.'
      : 'The local index uses TF-IDF tokenization and latent dimensions when available.'));
    $('model').replaceChildren(element('strong', 'Index ready · ' + book.model),
      element('p', book.dimensions + ' dimensions · ' + book.chunkCount + ' indexed chunks'), details);
    $('chunks').replaceChildren(element('p', 'Showing ' + book.preview.length + ' of ' + book.chunkCount + ' chunks.'),
      ...book.preview.map(chunk => sourceCard(chunk, chunk.text, chunk.text.length + ' characters')));
    $('vectors').replaceChildren(...book.preview.map(chunk => sourceCard(chunk, chunk.text,
      chunk.text.length + ' characters · ' + book.dimensions + ' dimensions · '
      + '[' + chunk.vector.map(value => value.toFixed(4)).join(', ')
      + (book.dimensions > chunk.vector.length ? ', …' : '') + ']', 'embedding-plot-' + chunk.id)));
    drawPlot($('embedding-plot'), book.preview);

}
async function refreshLibrary() {
  const saved = await request('/library');
  $('saved-books').replaceChildren(element('option', 'Choose a saved textbook'),
    ...saved.map(item => {
      const option = element('option', item.name);
      option.value = item.id;
      return option;
    }));
  $('library-label').hidden = !saved.length;
}
$('saved-books').onchange = async () => {
  const id = $('saved-books').value;
  if (!id) return;
  try {
    showBook(await request('/books/' + id));
    searchButton.disabled = false;
  } catch (error) {
    $('index-status').textContent = error.message;
  }
};
refreshLibrary().catch(() => {});

$('check-ollama').onclick = async () => {
  const button = $('check-ollama');
  button.disabled = true;
  $('ollama-status').textContent = 'Checking local Ollama and Nemotron…';
  try {
    browserLibrary ||= await import('/browser-library.js?v=evidence-ids-v4');
    const model = await browserLibrary.checkOllama();
    $('answer-mode').querySelector('[value=generated]').disabled = false;
    $('answer-mode').value = 'generated';
    $('ollama-status').textContent = model + ' is ready. Generated explanations enabled.';
  } catch (error) {
    $('ollama-status').textContent = 'Could not enable local generation. ' + error.message
      + ' Check that Ollama is running, the model is installed, and this site has local network permission.';
  } finally { button.disabled = false; }
};
