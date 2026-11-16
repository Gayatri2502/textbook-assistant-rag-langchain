const $ = id => document.getElementById(id);
const chat = $('chat');
const input = $('question');
const history = [];
const welcome = $('welcome').cloneNode(true);
let useNotes = false;

async function request(url, options = {}, timeout = 25000) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(timeout) });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Check your input and try again.');
  return body;
}

function line(role, text) {
  $('welcome')?.remove();
  const message = document.createElement('div');
  message.className = 'message ' + role;
  const label = document.createElement('strong');
  label.textContent = role === 'user' ? 'You' : useNotes ? 'My notes' : 'TravelAI';
  message.append(label, document.createTextNode(text));
  chat.append(message);
  return message;
}

function showSources(items) {
  for (const source of items) {
    const details = document.createElement('details');
    details.className = 'source';
    const summary = document.createElement('summary');
    summary.textContent = '[' + source.id + '] ' + source.source;
    const excerpt = document.createElement('p');
    excerpt.textContent = source.content;
    details.append(summary, excerpt);
    chat.append(details);
  }
}

function searchNotes(question) {
  const stop = new Set('the are was can how what when where which this that and for with from about should does will would have you your our tell find please my'.split(' '));
  const words = text => new Set((text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []).filter(word => word.length > 2 && !stop.has(word)));
  const query = words(question);
  const passages = $('notes-text').value.trim().split(/\n+|(?<=[.!?])\s+(?=\p{Lu})/u);
  const matches = passages
    .map(text => ({text: text.trim(), score: [...words(text)].filter(word => query.has(word)).length}))
    .filter(passage => passage.score > 0)
    .sort((a,b) => b.score - a.score)
    .slice(0, Number($('top-k').value));
  return {
    answer: matches.length ? matches.map((entry, i) => entry.text + ' [' + (i + 1) + ']').join('\n\n')
      : 'No matching passage in your trip notes. Try a specific term that appears in the notes, such as the hotel name or check-in.',
    sources: matches.map((entry, i) => ({ id: i + 1, source: 'My trip notes', content: entry.text }))
  };
}

function setBusy(busy) {
  for (const button of [$('ask'), $('clear'), $('use-notes'), $('use-docs'), ...document.querySelectorAll('[data-prompt]')]) button.disabled = busy;
  chat.setAttribute('aria-busy', String(busy));
}

async function submit() {
  const question = input.value.trim();
  if (!question || busyRun) return;
  const city = $('weather-city').value.trim();
  if (city.length === 1) {
    $('chat-error').textContent = 'Use at least two characters for the optional weather city.';
    $('chat-error').hidden = false;
    return;
  }
  if (useNotes && !$('notes-text').value.trim()) {
    $('chat-error').textContent = 'Paste some trip notes first, or switch to indexed documents.';
    $('chat-error').hidden = false;
    return;
  }
  $('chat-error').hidden = true;
  $('chat-status').textContent = 'Working on your question…';
  const start = beginRun(useNotes ? 'Browser notes search' : 'Document question');
  try {
    let data;
    if (useNotes) {
      receive({type:'stage',stage:'input',state:'done',elapsed_ms:0,detail:{question,mode:'browser'}});
      receive({type:'stage',stage:'query',state:'skipped',detail:{reason:'Browser notes search uses the entered question.'}});
      receive({type:'stage',stage:'embedding',state:'skipped',detail:{reason:'Browser search matches words.'}});
      const started = performance.now();
      receive({type:'stage',stage:'retrieve',state:'running',detail:{provider:'This browser tab'}});
      data = searchNotes(question);
      const duration = performance.now() - started;
      receive({type:'stage',stage:'retrieve',state:'done',duration_ms:duration,
        detail:{provider:'This browser tab',matched:data.sources.length,
          matches:data.sources.map(item=>({rank:item.id,source:item.source,excerpt:item.content.slice(0,350)}))}});
      for (const stage of ['geocode','weather','context','generate']) {
        receive({type:'stage',stage,state:'skipped',detail:{reason:'Browser notes search returns matching passages directly.'}});
      }
      receive({type:'result',...data,mode:'browser'});
    } else {
      data = await readEvents('/chat/trace', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({question,history:history.slice(-20),top_k:Number($('top-k').value),weather_location:city || null})
      });
    }
    line('user', question);
    line('assistant', data.answer);
    showSources(data.sources);
    chat.scrollTop = chat.scrollHeight;
    addAnswerLink('#travel-chat', 'View answer and citations');
    if (!useNotes) history.push({role:'user',content:question}, {role:'assistant',content:data.answer.slice(0,10000)});
    if (input.value.trim() === question) input.value = '';
    completeRun(start);
  } catch (error) {
    setRunState('error', 'Run failed');
    $('chat-error').textContent = explainFailure(error);
    $('chat-error').hidden = false;
  } finally {
    finishRun();
    $('chat-status').textContent = '';
  }
}

$('form').addEventListener('submit', event => { event.preventDefault(); submit(); });
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); submit(); }
});
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
  input.value = button.dataset.prompt;
  submit();
}));
$('clear').addEventListener('click', () => {
  history.length = 0;
  chat.replaceChildren(welcome.cloneNode(true));
  $('chat-error').hidden = true;
  input.focus();
});
function selectNotes(enabled) {
  if (enabled && !$('notes-text').value.trim()) {
    $('notes-status').textContent = 'Paste itinerary or booking notes before switching.';
    return;
  }
  useNotes = enabled;
  $('weather-city').disabled = enabled;
  describePath();
  history.length = 0;
  chat.replaceChildren(welcome.cloneNode(true));
  $('notes-status').textContent = enabled ? 'Searching my trip notes in this tab.' : 'Searching indexed documents.';
  $('question').labels[0].textContent = enabled ? 'Ask about my trip notes' : 'Ask your travel documents';
  input.focus();
}
$('use-notes').onclick = () => selectNotes(true);
$('use-docs').onclick = () => selectNotes(false);
document.querySelectorAll('[data-city]').forEach(button => button.onclick = () => {
  $('location').value = button.dataset.city;
  $('weather').requestSubmit();
});
$('weather').addEventListener('submit', async event => {
  event.preventDefault();
  const location = $('location').value.trim();
  if (location.length < 2 || busyRun) return;
  const started = beginRun('Live weather lookup');
  const out = $('weather-result');
  out.textContent = 'Checking weather…';
  try {
    const data = await readEvents('/weather/trace?location=' + encodeURIComponent(location));
    out.textContent = data.location + '\n' + data.current.temperature_2m + ' ' + data.units.temperature_2m
      + ' · Wind ' + data.current.wind_speed_10m + ' ' + data.units.wind_speed_10m
      + '\nAs of ' + data.current.time + ' (' + data.timezone + ')';
    addAnswerLink('#travel-weather', 'View current conditions');
    completeRun(started);
  } catch (error) {
    setRunState('error', 'Weather lookup failed');
    out.textContent = explainFailure(error);
  } finally {
    finishRun();
  }
});

const labels = {
  connect:'Service setup', input:'Input', query:'Query', embedding:'Embedding', retrieve:'Retrieval',
  geocode:'Resolve city', weather:'Current weather', context:'Context', generate:'Model answer',
};
const explanations = {
  input:'The question and count of prior messages passed into this run.',
  query:'The search query. Cloud follow-ups can be rewritten by OpenAI to include their context.',
  embedding:'OpenAI converts the search query into a vector. Only the embedding model and vector length are displayed.',
  retrieve:'The selected store ranks and returns matching document chunks. Scores appear only when Pinecone returns them.',
  geocode:'Open-Meteo resolves the requested location before weather is fetched.',
  weather:'Open-Meteo returns current modelled conditions and the provider timestamp.',
  context:'The answer pipeline prepares retrieved passages and any available weather evidence.',
  generate:'LangChain sends the prepared context to OpenAI and receives a cited response.',
};
const nodes = new Map();
const latest = new Map();
const logEntries = [];
let selectedEvent = null;
let follow = true;
let busyRun = false;
let runWarning = false;
let configuration;
let activeLink;

function nodeText(node, event) {
  const names = {running:'Running',done:'Done',skipped:'Skipped',error:'Failed'};
  node.querySelector('.node-meta').textContent = event
    ? names[event.state] + (event.duration_ms == null ? '' : ' · ' + durationLabel(event.duration_ms))
    : 'Waiting';
  node.dataset.state = event?.state || 'waiting';
}

function createNodes(container, stages) {
  for (const stage of stages) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'node';
    button.setAttribute('aria-pressed','false');
    const title = document.createElement('span');
    title.className = 'node-title';
    title.textContent = labels[stage];
    const meta = document.createElement('span');
    meta.className = 'node-meta';
    button.append(title, meta);
    nodeText(button);
    button.addEventListener('click', () => {
      setFollow(false);
      inspect(latest.get(stage) || {type:'stage',stage,state:'waiting'});
    });
    nodes.set(stage, button);
    $(container).append(button);
  }
}

function durationLabel(ms) {
  return ms < 1000 ? Number(ms).toFixed(ms < 10 ? 1 : 0) + ' ms' : (ms / 1000).toFixed(2) + ' s';
}

function describePath() {
  if (useNotes) {
    $('provider-path').textContent = 'This tab · Keyword search against your trip notes. OpenAI, Pinecone and live weather are outside this mode.';
  } else if (configuration?.mode === 'cloud' && configuration.cloud_configured) {
    const providers = configuration.providers;
    $('provider-path').textContent = 'Cloud configuration · LangChain → OpenAI embeddings → Pinecone '
      + providers.index + ' / ' + providers.namespace + ' → ' + providers.model;
  } else if (configuration?.mode === 'cloud') {
    $('provider-path').textContent = 'Cloud mode is missing provider configuration. Check OpenAI, Pinecone and the index name.';
  } else if (configuration?.mode === 'demo') {
    $('provider-path').textContent = 'Local document search · Cloud steps are skipped. Configure OpenAI and Pinecone to view the vector and model steps.';
  } else {
    $('provider-path').textContent = 'Provider configuration unavailable. Run a question to inspect the actual service status.';
  }
}

function setFollow(enabled) {
  follow = enabled;
  $('follow-live').setAttribute('aria-pressed', String(enabled));
}
function inspect(event) {
  if (!event) return;
  selectedEvent = event;
  const title = labels[event.stage] || (event.type === 'result' ? 'Run result' : event.type === 'warning' ? 'Warning' : 'Error');
  const elapsed = event.elapsed_ms == null ? '' : ' · +' + durationLabel(event.elapsed_ms);
  $('event-description').textContent = title + ' · ' + (event.state || event.type) + elapsed
    + (explanations[event.stage] ? '. ' + explanations[event.stage] : '');
  const detail = event.type === 'result'
    ? (typeof event.answer === 'string' ? event.answer : event.location + ' · ' + event.current?.time)
    : event.detail ? JSON.stringify(event.detail, null, 2) : event.message || '';
  $('event-data').textContent = detail;
  $('event-data').hidden = !detail;
  for (const [stage, button] of nodes) button.setAttribute('aria-pressed', String(stage === event.stage));
  for (const [row, button] of logEntries) button.setAttribute('aria-pressed', String(event === row));
}
function showMatches(detail) {
  $('matches').replaceChildren();
  $('match-count').textContent = detail.matched + (detail.top_k ? ' / ' + detail.top_k + ' requested' : ' matched');
  $('match-caption').textContent = detail.score_type
    ? 'Pinecone result order and raw scores. Scores are not percentages.'
    : detail.matched ? 'Matched excerpts in retrieval order.' : 'No documents matched the query.';
  for (const match of detail.matches || []) {
    const row = document.createElement('li');
    const details = document.createElement('details');
    const title = document.createElement('summary');
    title.textContent = '[' + match.rank + '] ' + match.source;
    if (Number.isFinite(match.score)) {
      const score = document.createElement('span');
      score.className = 'rank-score';
      score.textContent = 'Score ' + match.score.toFixed(4);
      title.append(score);
    }
    const excerpt = document.createElement('p');
    excerpt.textContent = match.excerpt;
    details.append(title,excerpt);
    row.append(details);
    $('matches').append(row);
  }
}
function showVector(detail) {
  const values = detail.sample || [];
  if (!values.length) return;
  const peak = Math.max(...values.map(value => Math.abs(value || 0)), 0.000001);
  $('vector-caption').textContent = 'First ' + values.length + ' of ' + detail.dimensions
    + ' coordinates. Bars use the largest displayed magnitude as the scale.';
  $('vector-plot').replaceChildren(...values.map((value,index) => {
    const bar = document.createElement('span');
    bar.dataset.sign = value < 0 ? 'negative' : 'positive';
    bar.style.setProperty('--magnitude', String(Math.abs(value || 0) / peak));
    bar.title = 'Coordinate ' + (index + 1) + ': ' + (value ?? 'unavailable');
    return bar;
  }));
  $('vector-details').hidden = false;
}
function receive(event) {
  if (event.type === 'stage') {
    latest.set(event.stage,event);
    if (nodes.has(event.stage)) nodeText(nodes.get(event.stage), event);
    if (event.stage === 'retrieve' && event.state === 'done') showMatches(event.detail);
    if (event.stage === 'embedding' && event.state === 'done') showVector(event.detail);
  }
  if (event.type === 'warning') runWarning = true;
  const item = document.createElement('li');
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'log-item';
  button.setAttribute('aria-pressed','false');
  button.textContent = (labels[event.stage] || event.type) + (event.state ? ' · ' + event.state : '')
    + (event.elapsed_ms == null ? '' : ' · +' + durationLabel(event.elapsed_ms));
  button.onclick = () => { setFollow(false); inspect(event); };
  item.append(button);
  logEntries.push([event,button]);
  $('timeline').append(item);
  $('event-count').textContent = '(' + logEntries.length + ')';
  if (follow) {
    inspect(event);
    $('timeline').scrollTop = $('timeline').scrollHeight;
  } else if (selectedEvent?.stage && selectedEvent.stage === event.stage
             && selectedEvent.state === 'waiting') {
    inspect(event);
  }
}

async function readEvents(url, options = {}) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(120000) });
  if (!response.ok) throw new Error(response.status === 422 ? 'Check the question and run options.' : 'The live pipeline endpoint is unavailable.');
  if (!response.body) throw new Error('The response did not include a live event stream.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result;
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream:!done});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      if (done && buffer.trim()) lines.push(buffer);
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        receive(event);
        if (event.type === 'error') throw new Error(event.message);
        if (event.type === 'result') result = event;
      }
      if (done) break;
    }
    if (!result) throw new Error('The event stream ended before returning a result.');
    return result;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function addAnswerLink(target, label) {
  activeLink = document.createElement('a');
  activeLink.href = target;
  activeLink.textContent = label;
  activeLink.className = 'pipeline-result-link';
  $('run-state').after(activeLink);
}
function setRunState(state,label) {
  $('run-state').dataset.state = state;
  $('run-state').textContent = label;
}
function beginRun(label) {
  busyRun = true;
  runWarning = false;
  setBusy(true);
  $('weather').querySelectorAll('button').forEach(button => button.disabled = true);
  $('top-k').disabled = true;
  $('weather-city').disabled = true;
  activeLink?.remove();
  latest.clear();
  logEntries.length = 0;
  selectedEvent = null;
  $('timeline').replaceChildren();
  $('timeline-details').open = true;
  $('matches').replaceChildren();
  $('vector-plot').replaceChildren();
  $('vector-details').hidden = true;
  $('match-count').textContent = '';
  $('match-caption').textContent = 'Matches appear here when retrieval completes.';
  $('event-count').textContent = '(0)';
  $('event-data').hidden = true;
  $('event-description').textContent = 'Waiting for the first event…';
  for (const button of nodes.values()) {
    nodeText(button);
    button.setAttribute('aria-pressed','false');
  }
  setFollow(true);
  setRunState('running', label + ' · Running');
  $('pipeline').scrollIntoView({block:'nearest'});
  return performance.now();
}
function completeRun(started) {
  setRunState('done', (runWarning ? 'Completed with warning' : 'Completed') + ' · ' + durationLabel(performance.now()-started));
}
function finishRun() {
  busyRun = false;
  setBusy(false);
  $('weather').querySelectorAll('button').forEach(button => button.disabled = false);
  $('top-k').disabled = false;
  $('weather-city').disabled = useNotes;
}
function explainFailure(error) {
  return error.name === 'TimeoutError' ? 'The request timed out. You can retry.'
    : 'Could not complete the request. ' + error.message;
}

createNodes('rag-nodes',['input','query','embedding','retrieve','context','generate']);
createNodes('weather-nodes',['geocode','weather']);
$('follow-live').onclick = () => {
  setFollow(!follow);
  if (follow && logEntries.length) inspect(logEntries.at(-1)[0]);
};
request('/status').then(data => {
  configuration = data;
  describePath();
  $('mode').textContent = data.mode === 'demo' ? 'Local document search'
    : data.cloud_configured ? 'Cloud · OpenAI + Pinecone' : 'Cloud configuration needed';
  $('source-description').textContent = data.mode === 'demo'
    ? 'Local answers quote matching passages. The bundled files are fictional samples.'
    : 'Cloud answers cite passages from your Pinecone document index.';
  for (const name of data.documents || ['Pinecone document index']) {
    const li = document.createElement('li');
    li.textContent = name;
    $('documents').append(li);
  }
}).catch(() => {
  describePath();
  $('mode').textContent = 'Document status unavailable';
  $('source-description').textContent = 'Reload to retry, or use My trip notes to search in this tab.';
});
