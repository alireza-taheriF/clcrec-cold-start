const $ = (id) => document.getElementById(id);

async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "request failed");
  return data;
}

function pct(x) {
  return `${Math.round(x * 1000) / 10}٪`;
}

function renderKpis(health, compare) {
  const b40 = (compare.launch || []).filter((r) => r.budget === 40);
  const first = b40.find((r) => r.policy === "firstslot");
  const rand = b40.find((r) => r.policy === "random");
  $("kpis").innerHTML = `
    <div class="kpi"><b>${health.n_new ?? "—"}</b><span>کد فنی صفر‌فروش</span></div>
    <div class="kpi"><b>${health.n_plants || health.n_users || "—"}</b><span>حساب خریدار / کارخانه</span></div>
    <div class="kpi"><b>${first ? pct(first.precision) : "—"}</b><span>دقت ۴۰ نمایش اول (FirstSlot)</span></div>
    <div class="kpi"><b>${rand && first ? `${Math.round((first.precision / Math.max(rand.precision, 1e-6)) * 10) / 10}×` : "—"}</b><span>بهتر از پاشش تصادفی</span></div>
  `;
}

function renderCompare(compare) {
  const b40 = (compare.launch || []).filter((r) => r.budget === 40);
  const labels = {
    random: "پاشش تصادفی",
    popular: "حساب‌های پرحجم",
    category: "فیلتر دستهٔ ERP",
    firstslot: "لانچ‌اسلات از مشخصات",
  };
  const max = Math.max(...b40.map((r) => r.precision), 0.01);
  $("compare").innerHTML = b40.map((r) => `
    <article class="bar-card ${r.policy === "firstslot" ? "win" : ""}">
      <h3>${labels[r.policy] || r.policy}</h3>
      <div class="bar"><i style="width:${(r.precision / max) * 100}%"></i></div>
      <small>دقت بودجه ۴۰: ${pct(r.precision)} · پوشش خریدار: ${pct(r.recall)}</small>
    </article>
  `).join("");
}

function renderSkus(items) {
  $("skus").innerHTML = items.map((it, i) => `
    <li data-id="${it.item_id}" class="${i === 0 ? "active" : ""}">
      <span class="sku">${it.sku}</span>
      <strong>${it.title}</strong>
      <em>${it.category_fa} · ${it.brand}</em>
    </li>
  `).join("");
}

function renderPlan(plan) {
  const item = plan.item || {};
  const audience = (plan.audience || []).map((u) => `
    <tr>
      <td>${u.rank}</td>
      <td>
        <strong>${u.name}</strong><br />
        <span class="why">${u.segment_fa || u.segment}</span>
      </td>
      <td>${u.affinity}</td>
      <td class="why">${(u.why || []).map((w) => w.title).join(" · ") || "—"}</td>
    </tr>
  `).join("");
  const subs = (plan.substitutes || []).map((s) => `
    <div class="card">
      <h3>${s.sku || s.item_id}</h3>
      <div>${s.title}</div>
      <div class="why">جایگذاری کنار این کد · امتیاز ${s.score}</div>
    </div>
  `).join("");

  $("result").innerHTML = `
    <div class="grid">
      <div>
        <h3>${item.title || "برنامهٔ لانچ"}</h3>
        <p class="why">بودجه ${plan.budget} نمایش · ${plan.n_scored_users} حساب امتیاز گرفتند</p>
        <table>
          <thead><tr><th>#</th><th>حساب</th><th>قرابیت</th><th>چرا این کارخانه</th></tr></thead>
          <tbody>${audience}</tbody>
        </table>
      </div>
      <div>
        <h3>جایگزین / جایگذاری</h3>
        ${subs || "<p class='why'>جایگزین گرم پیدا نشد.</p>"}
      </div>
    </div>
  `;
}

async function planSelected(itemId) {
  $("status").textContent = "در حال تخصیص بودجه…";
  const plan = await postJson("/v1/launch/plan", {
    item_id: Number(itemId),
    budget: Number($("budget").value),
    diversity: Number($("diversity").value) / 100,
  });
  $("status").textContent = "بودجه تخصیص داده شد. این فهرست را به تیم فروش/بازرگانی بدهید.";
  renderPlan(plan);
}

async function planNew() {
  $("status").textContent = "کد خارج از کاتالوگ در حال embed…";
  const plan = await postJson("/v1/launch/plan", {
    title: $("new-title").value,
    tags: $("new-tags").value,
    budget: Number($("budget").value),
    diversity: Number($("diversity").value) / 100,
  });
  $("status").textContent = "کد جدید بدون هیچ سفارشی پرتاب شد.";
  renderPlan(plan);
}

async function boot() {
  const [health, compare, items] = await Promise.all([
    getJson("/v1/health"),
    getJson("/v1/launch/compare"),
    getJson("/v1/launch/items"),
  ]);
  renderKpis(health, compare);
  renderCompare(compare);
  renderSkus(items.items || []);
  const first = (items.items || [])[0];
  if (first) await planSelected(first.item_id);

  $("skus").addEventListener("click", (ev) => {
    const li = ev.target.closest("li[data-id]");
    if (!li) return;
    document.querySelectorAll(".skus li").forEach((n) => n.classList.remove("active"));
    li.classList.add("active");
    planSelected(li.dataset.id);
  });
  $("run").addEventListener("click", () => {
    const active = document.querySelector(".skus li.active");
    if (active) planSelected(active.dataset.id);
  });
  $("run-new").addEventListener("click", planNew);
}

boot().catch((err) => {
  $("status").textContent = `خطا: ${err.message}. اول python main.py --product را اجرا کنید.`;
});
