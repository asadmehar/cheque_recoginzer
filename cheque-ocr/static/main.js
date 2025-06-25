document.getElementById('run').addEventListener('click', async () => {
  const fileInput = document.getElementById('file');
  if (!fileInput.files.length) return alert('Choose an image first');

  const fd = new FormData();
  fd.append('file', fileInput.files[0]);

  document.getElementById('out').textContent = 'Processing…';

  const res = await fetch('/predict', { method: 'POST', body: fd });
  const json = await res.json();

  document.getElementById('out').textContent =
    JSON.stringify(json, null, 2);
});
