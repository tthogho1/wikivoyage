async function postChat(question, top_k=5){
  const body = {question, top_k};
  const resp = await fetch('/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  if(!resp.ok){
    const txt = await resp.text();
    throw new Error(`Server error ${resp.status}: ${txt}`);
  }
  return resp.json();
}

function formatSource(s){
  return `<div class="source"><strong>${s.title}</strong> <div class="meta">${s.page_type} ${s.status?`• ${s.status}`:''} ${s.url?`• <a href="${s.url}" target="_blank">link</a>`:''}</div><pre style="white-space:pre-wrap">${s.retrieval_text}</pre></div>`;
}

window.addEventListener('DOMContentLoaded', ()=>{
  const btn = document.getElementById('send');
  const clear = document.getElementById('clear');
  const prompt = document.getElementById('prompt');
  const topk = document.getElementById('top_k');
  const status = document.getElementById('status');
  const answer = document.getElementById('answer');
  const sources = document.getElementById('sources');

  btn.addEventListener('click', async ()=>{
    const q = prompt.value.trim();
    if(!q){ return; }
    status.textContent = 'Thinking…';
    answer.style.display='none';
    sources.innerHTML='';
    try{
      const res = await postChat(q, parseInt(topk.value||5,10));
      answer.innerHTML = res.answer.replace(/\n/g,'<br>');
      answer.style.display='block';
      if(res.sources && res.sources.length){
        res.sources.forEach(s=>{ sources.innerHTML += formatSource(s); });
      }
      status.textContent = '';
    }catch(err){
      status.textContent = 'Error: ' + err.message;
    }
  });

  clear.addEventListener('click', ()=>{ prompt.value=''; answer.style.display='none'; sources.innerHTML=''; status.textContent=''; });
});
