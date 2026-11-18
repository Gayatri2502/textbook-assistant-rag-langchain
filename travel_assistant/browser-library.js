let chunks = [];
const words = text => text.toLowerCase().match(/[a-z]{3,}/g) || [];
const stop = new Set('the and for are with what does how why this that from into have which'.split(' '));
export async function index(file, settings, progress) {
  const pdfjs = await import('https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs');
  pdfjs.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs';
  const pdf = await pdfjs.getDocument({data:await file.arrayBuffer()}).promise;
  if (pdf.numPages > 5000) { await pdf.destroy(); throw Error('Use a PDF with at most 5,000 pages.'); }
  const next = [];
  let textPages = 0;
  try {
    for (let page = 1; page <= pdf.numPages; page++) {
      const source = await pdf.getPage(page);
      const content = await source.getTextContent();
      const text = content.items.map(item => item.str + (item.hasEOL ? '\n' : ' ')).join('');
      if (text.trim()) textPages++;
      const size = Number(settings.get('size')), overlap = Number(settings.get('overlap'));
      for (let start = 0; start < text.length;) {
        let end = Math.min(start + size, text.length);
        if (settings.get('strategy') !== 'fixed' && end < text.length) {
          const separators = settings.get('separators').split('|').filter(Boolean).map(value => value.replaceAll('\\n', '\n'));
          for (const separator of separators) {
            const boundary = text.lastIndexOf(separator, end - separator.length);
            if (boundary > start + Math.max(overlap, size / 2)) {
              end = boundary + separator.length;
              break;
            }
          }
        }
        const passage = text.slice(start, end);
        next.push({id:next.length+1,page,text:passage,excerpt:passage,vector:[],x:page,y:next.length});
        if (next.length > 100000) throw Error('Too many passages. Increase chunk size.');
        if (end === text.length) break;
        start = end - overlap;
      }
      source.cleanup();
      progress(page, pdf.numPages);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (!next.length) throw Error('No selectable text found. OCR is required for scanned books.');
    chunks = next;
    return {id:'browser',name:file.name,pages:pdf.numPages,textPages,chunkCount:chunks.length,
      model:'Browser keyword search',dimensions:0,preview:chunks.filter((_,i)=>i % Math.ceil(chunks.length/120)===0),
      pdfUrl:URL.createObjectURL(file)};
  } finally { await pdf.destroy(); }
}
export function search(question, limit) {
  const query = [...new Set(words(question).filter(word=>!stop.has(word)))];
  const matches = chunks.map(chunk => {
    const text = new Set(words(chunk.text));
    return {...chunk,score:query.filter(word=>text.has(word)).length / Math.max(query.length,1)};
  }).filter(chunk=>chunk.score>0).sort((a,b)=>b.score-a.score).slice(0,limit);
  return {matches,point:[0,0],searched:chunks.length,score_type:'Term coverage',query_vector:[],dimensions:0};
}
export function answer(result) {
  return {mode:'extractive',claims:result.matches.map(match=>({text:match.text,quote:match.text,
    excerpt:match.text,offset:0,page:match.page,chunk:match.id})),unmatched_terms:[]};
}
