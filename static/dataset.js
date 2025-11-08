// Dataset page logic: upload XLSX files, list existing files with metadata, and provide download/edit links

function formatDate(ts) {
  try {
    const d = new Date(ts * 1000);
    return d.toLocaleString();
  } catch (e) {
    return '--';
  }
}

async function fetchDatasets() {
  const res = await fetch('/api/datasets');
  if (!res.ok) throw new Error('Falha ao listar datasets');
  return await res.json();
}

function renderDatasets(items) {
  const tbody = document.getElementById('dataset-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';
  if (!items || items.length === 0) {
    tbody.innerHTML = '<tr><td class="px-6 py-3" colspan="6">Nenhum dataset enviado ainda.</td></tr>';
    return;
  }
  items.forEach(item => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="px-6 py-3">${item.display_name || (item.name || '').replace(/\.(csv|xlsx)$/i,'')}</td>
      <td class="px-6 py-3">${item.size || '--'}</td>
      <td class="px-6 py-3">${item.rows ?? '--'}</td>
      <td class="px-6 py-3">${formatDate(item.created_at)}</td>
      <td class="px-6 py-3">${formatDate(item.updated_at)}</td>
      <td class="px-6 py-3 flex items-center gap-2">
        <a class="inline-flex items-center justify-center p-2 rounded text-blue-600 hover:bg-gray-100 dark:hover:bg-[#1f2d3a]" href="${item.download_url}" aria-label="Baixar" title="Baixar">
          <span class="material-symbols-outlined">download</span>
        </a>
        <a class="inline-flex items-center justify-center p-2 rounded text-green-600 hover:bg-gray-100 dark:hover:bg-[#1f2d3a]" href="${item.edit_url}" aria-label="Editar" title="Editar">
          <span class="material-symbols-outlined">edit</span>
        </a>
        <button class="inline-flex items-center justify-center p-2 rounded text-red-600 hover:bg-gray-100 dark:hover:bg-[#1f2d3a]" data-delete="${item.name}" aria-label="Excluir" title="Excluir">
          <span class="material-symbols-outlined">delete</span>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function humanReadableBytes(bytes) {
  if (bytes == null) return '--';
  const units = ['B','KB','MB','GB','TB'];
  let size = Number(bytes);
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx++;
  }
  return `${size.toFixed(2)} ${units[idx]}`;
}

function updateStats(items) {
  const countEl = document.getElementById('stat-count');
  const rowsEl = document.getElementById('stat-rows');
  const sizeEl = document.getElementById('stat-size');
  const updEl = document.getElementById('stat-updated');
  if (!countEl || !rowsEl || !sizeEl || !updEl) return;
  const count = items?.length || 0;
  const totalRows = (items || []).reduce((acc, it) => acc + (typeof it.rows === 'number' ? it.rows : 0), 0);
  const totalBytes = (items || []).reduce((acc, it) => acc + (typeof it.size_bytes === 'number' ? it.size_bytes : 0), 0);
  const lastUpdatedTs = (items || []).reduce((acc, it) => Math.max(acc, it.updated_at || 0), 0);
  countEl.textContent = String(count);
  rowsEl.textContent = String(totalRows);
  sizeEl.textContent = humanReadableBytes(totalBytes);
  updEl.textContent = lastUpdatedTs ? formatDate(lastUpdatedTs) : '--';
}

async function uploadCSV(file, datasetName) {
  const fd = new FormData();
  fd.append('file', file);
  if (datasetName) {
    fd.append('dataset_name', datasetName);
  }
  const res = await fetch('/api/datasets/upload', { method: 'POST', body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || 'Falha no upload');
  }
  return await res.json();
}

async function deleteDataset(filename) {
  const res = await fetch(`/api/datasets/delete/${encodeURIComponent(filename)}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || 'Falha ao excluir dataset');
  }
  return await res.json();
}

function bindEvents() {
  const uploadBtn = document.getElementById('upload-btn');
  const fileInput = document.getElementById('dataset-file');
  const nameInput = document.getElementById('dataset-name');
  const refreshBtn = document.getElementById('refresh-btn');
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener('click', async () => {
      const file = fileInput.files && fileInput.files[0];
      const datasetName = nameInput ? nameInput.value.trim() : '';
      if (!datasetName) {
        alert('Informe o nome do dataset.');
        return;
      }
      if (!file) {
        alert('Selecione um arquivo .xlsx para enviar.');
        return;
      }
      try {
        await uploadCSV(file, datasetName);
        const items = await fetchDatasets();
        renderDatasets(items);
        updateStats(items);
        fileInput.value = '';
        if (nameInput) nameInput.value = '';
      } catch (e) {
        alert(e.message);
      }
    });
  }
  // Desabilitar o botão se campos obrigatórios não estiverem preenchidos
  function updateUploadButtonState() {
    const hasFile = !!(fileInput && fileInput.files && fileInput.files[0]);
    const hasName = !!(nameInput && nameInput.value.trim());
    if (uploadBtn) uploadBtn.disabled = !(hasFile && hasName);
  }
  if (fileInput) fileInput.addEventListener('change', updateUploadButtonState);
  if (nameInput) nameInput.addEventListener('input', updateUploadButtonState);
  updateUploadButtonState();
  if (refreshBtn) {
    refreshBtn.addEventListener('click', async () => {
      try {
        const items = await fetchDatasets();
        renderDatasets(items);
        updateStats(items);
      } catch (e) {
        alert('Falha ao atualizar lista: ' + e.message);
      }
    });
  }
  // Delegação de eventos para excluir datasets
  const tbody = document.getElementById('dataset-table-body');
  if (tbody) {
    tbody.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-delete]');
      if (!btn) return;
      const fname = btn.getAttribute('data-delete');
      const ok = confirm(`Excluir o dataset "${fname}"?`);
      if (!ok) return;
      try {
        await deleteDataset(fname);
        const items = await fetchDatasets();
        renderDatasets(items);
        updateStats(items);
      } catch (err) {
        alert(err.message);
      }
    });
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  try {
    const items = await fetchDatasets();
    renderDatasets(items);
    updateStats(items);
  } catch (e) {
    // Mostrar mensagem mínima
    const tbody = document.getElementById('dataset-table-body');
    if (tbody) {
      tbody.innerHTML = `<tr><td class="px-6 py-3" colspan="6">Erro ao carregar datasets: ${e.message}</td></tr>`;
    }
  }
});