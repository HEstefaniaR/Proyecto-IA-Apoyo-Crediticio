// ═══════════════════════════════════════════════════════
//  CONFIGURACIÓN
// ═══════════════════════════════════════════════════════
const API_URL = (window.APP_CONFIG?.apiUrl || 'http://localhost:5001').replace(/\/$/, '');

// ── Estado global ──────────────────────────────────────
let todosClientes   = [];
let clientesData    = {};
let cerradosSet     = new Set(JSON.parse(localStorage.getItem('cerrados') || '[]'));
let mostrarCerrados = false;
let clienteActivo   = null;
let asesoresMap     = JSON.parse(localStorage.getItem('asesoresMap') || '{}');
let filtroAsesor    = 'all';
const ASESORES      = ['Maryory Rojas', 'Nicol Pérez'];
let cerradosMotivo  = JSON.parse(localStorage.getItem('cerradosMotivo') || '{}');

// ═══════════════════════════════════════════════════════
//  CARGA DE CLIENTES
// ═══════════════════════════════════════════════════════
async function cargarClientes() {
  const container = document.getElementById('cards-container');
  container.innerHTML = `
    <div class="loader-wrap">
      <div class="spinner"></div>
      <div class="loader-text">Descargando clientes desde Botpress…</div>
    </div>`;

  try {
    // Cargar asesores del servidor y hacer merge con localStorage
    try {
      const resA = await fetch(`${API_URL}/api/asesores`);
      const dataA = await resA.json();
      const asesoresServidor = dataA.asesores || {};
      asesoresMap = { ...asesoresMap, ...asesoresServidor };
    } catch(e) {
      console.warn('No se pudo cargar asesores del servidor, usando localStorage:', e);
    }

    const res = await fetch(`${API_URL}/api/perfilar_cliente`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`API ${res.status}: ${err}`);
    }

    const data = await res.json();

    if (!data.clientes || data.clientes.length === 0) {
      container.innerHTML = `<div class="empty-state"><div class="empty-icon">📭</div>No hay clientes en Botpress todavía.</div>`;
      actualizarStats([]);
      return;
    }

    todosClientes = data.clientes;

    todosClientes.forEach(c => {
      if (c.prediccion) clientesData[String(c.cedula)] = c.prediccion;
    });

    const cerradosServidor = todosClientes
      .filter(c => c.cerrado)
      .map(c => String(c.cedula));
    cerradosServidor.forEach(ced => cerradosSet.add(ced));

    const lista = buildLista();
    renderTarjetas(lista);
    actualizarStats(lista);

  } catch (e) {
    container.innerHTML = `
      <div class="error-box">
        <strong>❌ Error al conectar con la API</strong>
        ${e.message}<br><br>
        Verifica que:<br>
        • La API está corriendo en <code>${API_URL}</code><br>
        • Las credenciales de Botpress son correctas<br>
        • Hay clientes en la tabla
      </div>`;
  }
}

// ═══════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════
function nivelColor(prob) {
  if (prob >= 70) return 'high';
  if (prob >= 40) return 'medium';
  return 'low';
}

function buildLista() {
  return todosClientes
    .map(c => ({ cliente: c, resultado: clientesData[String(c.cedula)] || null }))
    .filter(({ cliente }) => {
      if (filtroAsesor === 'all') return true;
      if (filtroAsesor === 'sin_asignar') return !asesoresMap[String(cliente.cedula)];
      return asesoresMap[String(cliente.cedula)] === filtroAsesor;
    })
    .sort((a, b) => (b.resultado?.prob_aprobacion ?? 0) - (a.resultado?.prob_aprobacion ?? 0));
}

// ═══════════════════════════════════════════════════════
//  RENDER TARJETAS
// ═══════════════════════════════════════════════════════
function renderTarjetas(lista, filtro = 'all', busqueda = '') {
  const container = document.getElementById('cards-container');
  container.innerHTML = '';

  const grid = document.createElement('div');
  grid.className = 'cards-grid';
  container.appendChild(grid);

  let contVisible = 0;

  lista.forEach(({ cliente, resultado }, idx) => {
    const prob      = resultado?.prob_aprobacion ?? 0;
    const nivel     = nivelColor(prob);
    const esCerrado = cerradosSet.has(String(cliente.cedula));

    if (filtro !== 'all' && nivel !== filtro) return;
    if (busqueda) {
      const q = busqueda.toLowerCase();
      if (!cliente.nombre?.toLowerCase().includes(q) && !String(cliente.cedula).includes(q)) return;
    }
    if (esCerrado && !mostrarCerrados) return;

    contVisible++;

    const card = document.createElement('div');
    card.className = `client-card${esCerrado ? ' cerrado visible' : ''}`;
    card.style.animationDelay = `${idx * 0.04}s`;
    card.dataset.cedula = cliente.cedula;

    const nombreDisplay = cliente.nombre?.toUpperCase() || 'SIN NOMBRE';
    const celular = cliente.celular || '';
    const monto   = resultado ? `$${Number(cliente.monto).toLocaleString('es-CO')}` : '—';
    const plazo   = resultado ? `${cliente.plazo} meses` : '—';
    const asesorNombre = asesoresMap[String(cliente.cedula)];

    card.innerHTML = `
      <div class="card-stripe ${nivel}"></div>
      <div class="card-body">
        <div class="card-header-row">
          <div>
            <div class="client-name">${nombreDisplay}</div>
            <div class="client-cedula">CC ${cliente.cedula}</div>
          </div>
          <div class="prob-badge">
            <div class="prob-pct ${nivel}">${prob.toFixed(0)}%</div>
            <div class="prob-label">Viabilidad</div>
          </div>
        </div>
        <div class="card-info-row">
          <span class="info-chip">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
              <line x1="12" y1="1" x2="12" y2="23"/>
              <path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/>
            </svg>
            ${monto}
          </span>
          <span class="info-chip">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
              <line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/>
              <line x1="3" y1="10" x2="21" y2="10"/>
            </svg>
            ${plazo}
          </span>
          ${esCerrado ? `<span class="info-chip" style="color:#16a34a;background:#dcfce7;">✓ ${cerradosMotivo[String(cliente.cedula)] === 'ganado' ? 'Ganado' : 'Cerrado'}</span>` : ''}
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar-bg">
            <div class="prob-bar-fill ${nivel}" style="width:${prob}%"></div>
          </div>
        </div>
        <div class="card-footer">
          <a class="phone-link" href="tel:${celular}">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
              <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"/>
            </svg>
            ${celular || 'Sin teléfono'}
          </a>
          <div style="display:flex;gap:.4rem;align-items:center;">
            ${asesorNombre ? `<span class="asesor-chip">${asesorNombre.split(' ')[0]}</span>` : ''}
            <button class="btn-ver" onclick="abrirModal('${cliente.cedula}')">Ver detalle</button>
          </div>
        </div>
      </div>`;

    grid.appendChild(card);
  });

  if (contVisible === 0) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1"><div class="empty-icon">🔍</div>No hay clientes con ese filtro.</div>';
  }
}

// ═══════════════════════════════════════════════════════
//  MODAL DETALLE
// ═══════════════════════════════════════════════════════
function abrirModal(cedula) {
  clienteActivo = todosClientes.find(c => String(c.cedula) === String(cedula));
  if (!clienteActivo) return;

  const resultado = clientesData[String(cedula)];
  document.getElementById('modal-nombre').textContent     = clienteActivo.nombre?.toUpperCase() || '—';
  document.getElementById('modal-cedula-sub').textContent = `CC ${cedula} · ${clienteActivo.pagaduria?.toUpperCase() || ''}`;

  const body = document.getElementById('modal-body');

  if (!resultado) {
    body.innerHTML = `<div class="error-box"><strong>Sin resultado del modelo</strong>No se pudo obtener el análisis para este cliente.</div>`;
  } else {
    const prob   = resultado.prob_aprobacion;
    const nivel  = nivelColor(prob);
    const hist   = resultado.historial_cliente || {};
    const rank   = resultado.ranking_cooperativas || [];
    const mejor  = resultado.mejor_opcion_elegible;

    body.innerHTML = `
      <div class="section-title">Datos del cliente</div>
      <div class="info-grid">
        <div class="info-item"><div class="info-key">Nombre</div><div class="info-val">${clienteActivo.nombre?.toUpperCase()}</div></div>
        <div class="info-item"><div class="info-key">Cédula</div><div class="info-val">${cedula}</div></div>
        <div class="info-item"><div class="info-key">Edad</div><div class="info-val">${clienteActivo.edad} años</div></div>
        <div class="info-item"><div class="info-key">Celular</div><div class="info-val">${clienteActivo.celular || '—'}</div></div>
        <div class="info-item"><div class="info-key">Pagaduría</div><div class="info-val">${clienteActivo.pagaduria?.toUpperCase() || '—'}</div></div>
        <div class="info-item"><div class="info-key">Tipo de crédito</div><div class="info-val">${clienteActivo.tipo_credito?.toUpperCase() || '—'}</div></div>
        <div class="info-item"><div class="info-key">Monto solicitado</div><div class="info-val">$${Number(clienteActivo.monto).toLocaleString('es-CO')}</div></div>
        <div class="info-item"><div class="info-key">Plazo</div><div class="info-val">${clienteActivo.plazo} meses</div></div>
      </div>

      <div class="section-title">Historial en BD</div>
      <div class="historial-chips">
        <div class="h-chip">
          <div class="h-chip-label">Estado</div>
          <div class="h-chip-val" style="font-size:.85rem;">${hist.es_cliente_nuevo ? '🆕 Nuevo' : '✅ Antiguo'}</div>
        </div>
        <div class="h-chip">
          <div class="h-chip-label">Créditos previos</div>
          <div class="h-chip-val">${hist.num_creditos_totales ?? '—'}</div>
        </div>
        <div class="h-chip">
          <div class="h-chip-label">Días desde último</div>
          <div class="h-chip-val">${hist.dias_desde_ultimo_credito >= 9999 ? '—' : hist.dias_desde_ultimo_credito + 'd'}</div>
        </div>
      </div>

      <div class="section-title">Modelo 1 — Viabilidad de aprobación</div>
      <div class="m1-block">
        <div class="m1-circle ${nivel}">
          <div class="m1-pct ${nivel}">${prob.toFixed(1)}%</div>
          <div class="m1-sub">viab.</div>
        </div>
        <div class="m1-desc">
          <h4>${nivel === 'high' ? 'Alta viabilidad' : nivel === 'medium' ? 'Viabilidad media' : 'Baja viabilidad'}</h4>
          <p>${
            nivel === 'high'
              ? 'El perfil del cliente tiene alta probabilidad de aprobación. Priorizar gestión.'
              : nivel === 'medium'
              ? 'Perfil con viabilidad moderada. Evaluar cooperativas disponibles.'
              : 'Perfil con baja probabilidad. Revisar condiciones y posibles alternativas.'
          }</p>
          ${mejor ? `<p style="margin-top:.4rem;font-size:.75rem;color:var(--azul);">✦ Mejor opción elegible: <strong>${mejor.cooperativa}</strong> (${mejor.prob_ml}% confianza)</p>` : ''}
        </div>
      </div>

      <div class="section-title">Modelo 2 — Ranking de cooperativas</div>
      <div class="coop-list">
        ${rank.slice(0, 5).map((c, i) => `
          <div class="coop-item">
            <div class="coop-rank ${i === 0 ? 'gold' : ''}">${i + 1}</div>
            <div style="flex:1">
              <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;">
                <span class="coop-name">${c.cooperativa}</span>
                <span class="coop-elegible ${c.elegible_reglas ? 'ok' : 'no'}">${c.elegible_reglas ? '✓ Elegible' : '✗ No elegible'}</span>
              </div>
              ${!c.elegible_reglas && c.razones_rechazo?.length
                ? `<div class="coop-razones">${c.razones_rechazo.join(' · ')}</div>`
                : ''}
            </div>
            <div class="coop-bar-wrap"><div class="coop-bar-fill" style="width:${c.prob_ml}%"></div></div>
            <div class="coop-pct">${c.prob_ml}%</div>
          </div>`).join('')}
      </div>`;
  }

  // Footer dinámico
  const asesorActual = asesoresMap[String(cedula)] || '';
  document.getElementById('modal-footer-content').innerHTML = `
    <div class="asesor-select-wrap">
      <span class="asesor-select-label">Asesor:</span>
      <select id="select-asesor" class="asesor-select" onchange="asignarAsesor('${cedula}', this.value)">
        <option value="">— Sin asignar —</option>
        ${ASESORES.map(a => `<option value="${a}" ${asesorActual === a ? 'selected' : ''}>${a}</option>`).join('')}
      </select>
    </div>
    <a class="btn-wsp" href="https://wa.me/57${clienteActivo.celular || ''}" target="_blank">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347z"/><path d="M11.999 2C6.477 2 2 6.477 2 12c0 1.99.574 3.842 1.564 5.407L2 22l4.737-1.543A9.953 9.953 0 0012 22c5.523 0 10-4.477 10-10S17.523 2 11.999 2zm0 18a7.95 7.95 0 01-4.07-1.115l-.291-.174-3.019.983.899-3.049-.19-.311A7.96 7.96 0 014 12c0-4.411 3.589-8 8-8s8 3.589 8 8-3.589 8-8 8z"/></svg>
      WhatsApp
    </a>
    <button class="btn-cerrar-caso btn-ganado" onclick="cerrarCaso('ganado')">✓ Ganado</button>
    <button class="btn-cerrar-caso" onclick="cerrarCaso('perdido')">✗ Perdido</button>`;

  document.getElementById('modal-overlay').classList.add('open');
}

function cerrarModal() {
  document.getElementById('modal-overlay').classList.remove('open');
  clienteActivo = null;
}

function cerrarModalSiFondo(e) {
  if (e.target === document.getElementById('modal-overlay')) cerrarModal();
}

// ═══════════════════════════════════════════════════════
//  CERRAR CASO
// ═══════════════════════════════════════════════════════
async function cerrarCaso(motivo = 'perdido') {
  if (!clienteActivo) return;
  const cedula = String(clienteActivo.cedula);

  try {
    await fetch(`${API_URL}/api/cliente/${cedula}/cerrar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ motivo })
    });
  } catch (e) {
    console.warn('No se pudo notificar al servidor:', e);
  }

  cerradosSet.add(cedula);
  cerradosMotivo[cedula] = motivo;
  localStorage.setItem('cerrados', JSON.stringify([...cerradosSet]));
  localStorage.setItem('cerradosMotivo', JSON.stringify(cerradosMotivo));

  const card = document.querySelector(`.client-card[data-cedula="${cedula}"]`);
  if (card) {
    card.classList.add('cerrado');
    if (!mostrarCerrados) card.style.display = 'none';
  }

  actualizarStats(buildLista());
  cerrarModal();
}

// ═══════════════════════════════════════════════════════
//  FILTROS Y BÚSQUEDA
// ═══════════════════════════════════════════════════════
function filtrarTarjetas() {
  const filtro   = document.getElementById('filter-select').value;
  const busqueda = document.getElementById('search-input').value;
  renderTarjetas(buildLista(), filtro, busqueda);
}

function toggleCerrados() {
  mostrarCerrados = !mostrarCerrados;
  const btn = document.getElementById('toggle-cerrados');
  btn.classList.toggle('active', mostrarCerrados);
  btn.textContent = mostrarCerrados ? 'Ocultar cerrados' : 'Mostrar cerrados';
  filtrarTarjetas();
}

function filtrarPorAsesor(val) {
  filtroAsesor = val;
  renderTarjetas(buildLista(),
    document.getElementById('filter-select').value,
    document.getElementById('search-input').value);
}

// ═══════════════════════════════════════════════════════
//  STATS
// ═══════════════════════════════════════════════════════
function actualizarStats(lista) {
  const activos = lista.filter(({ cliente }) => !cerradosSet.has(String(cliente?.cedula || '')));
  const alta    = activos.filter(({ resultado }) => (resultado?.prob_aprobacion ?? 0) >= 70).length;
  const media   = activos.filter(({ resultado }) => {
    const p = resultado?.prob_aprobacion ?? 0;
    return p >= 40 && p < 70;
  }).length;

  const ganados  = [...cerradosSet].filter(c => cerradosMotivo[c] === 'ganado').length;
  const perdidos = [...cerradosSet].filter(c => cerradosMotivo[c] === 'perdido').length;

  document.getElementById('stat-cola').textContent     = activos.length;
  document.getElementById('stat-alta').textContent     = alta;
  document.getElementById('stat-media').textContent    = media;
  document.getElementById('stat-cerrados').textContent = cerradosSet.size;
  document.getElementById('badge-total').textContent   = `${activos.length} clientes activos`;

  const elG = document.getElementById('stat-ganados');
  const elP = document.getElementById('stat-perdidos');
  if (elG) elG.textContent = ganados;
  if (elP) elP.textContent = perdidos;
}

// ═══════════════════════════════════════════════════════
//  ASESOR
// ═══════════════════════════════════════════════════════
async function asignarAsesor(cedula, asesor) {
  asesoresMap[cedula] = asesor;
  localStorage.setItem('asesoresMap', JSON.stringify(asesoresMap));
  try {
    await fetch(`${API_URL}/api/cliente/${cedula}/asesor`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ asesor })
    });
  } catch(e) { console.warn('No se pudo guardar asesor en servidor:', e); }

  const card = document.querySelector(`.client-card[data-cedula="${cedula}"]`);
  if (card) {
    const footer = card.querySelector('.card-footer > div');
    if (footer) {
      const existingChip = footer.querySelector('.asesor-chip');
      if (existingChip) existingChip.remove();
      if (asesor) {
        const chip = document.createElement('span');
        chip.className = 'asesor-chip';
        chip.textContent = asesor.split(' ')[0];
        footer.insertBefore(chip, footer.firstChild);
      }
    }
  }
  actualizarStats(buildLista());
}

// ═══════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', cargarClientes);