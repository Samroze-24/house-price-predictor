// House Price Predictor — frontend logic (no build step, no API keys).
// Talks to the Flask backend: POST /predict and GET /api/city-averages.

function formatPKR(value) {
  return "PKR " + Math.round(value).toLocaleString("en-PK");
}

function formatCompact(value) {
  // 1 crore = 10,000,000 PKR; 1 lakh = 100,000 PKR.
  if (value >= 10000000) return (value / 10000000).toFixed(2) + " crore";
  if (value >= 100000) return (value / 100000).toFixed(1) + " lakh";
  return formatPKR(value);
}

function signedPKR(value) {
  const sign = value >= 0 ? "+" : "−";
  return sign + formatPKR(Math.abs(value));
}

async function loadCityAverages() {
  const container = document.getElementById("city-averages");
  try {
    const res = await fetch("/api/city-averages");
    const data = await res.json();
    container.innerHTML = data.cities
      .map(
        (c) =>
          `<div class="city-row"><span><strong>${c.city}</strong> ` +
          `<span class="hint">(${c.sample_count} sample houses)</span></span>` +
          `<strong>${formatPKR(c.average_price_pkr)}</strong></div>`
      )
      .join("");
  } catch {
    container.innerHTML = '<p class="error">Could not load city averages.</p>';
  }
}

document.getElementById("predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.target;
  const errorEl = document.getElementById("form-error");
  const submitBtn = document.getElementById("submit-btn");
  errorEl.hidden = true;
  submitBtn.disabled = true;
  submitBtn.textContent = "Estimating…";

  const payload = {
    city: form.city.value,
    area: form.area.value,
    area_unit: form.area_unit.value,
    bedrooms: form.bedrooms.value,
    bathrooms: form.bathrooms.value,
    age_years: form.age_years.value,
    location_tier: form.location_tier.value,
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || !data.success) {
      throw new Error(data.error || "Prediction failed.");
    }

    document.getElementById("price").textContent = formatPKR(data.estimated_price_pkr);
    document.getElementById("price-range").textContent =
      `Typical range (model MAE band): ${formatPKR(data.price_range_pkr.low)} – ` +
      `${formatPKR(data.price_range_pkr.high)}`;
    document.getElementById("price-compact").textContent =
      `About ${formatCompact(data.estimated_price_pkr)}`;

    const maxImpact = Math.max(...data.feature_impacts.map((i) => Math.abs(i.delta_pkr)), 1);
    document.getElementById("impacts").innerHTML = data.feature_impacts
      .map((item) => {
        const up = item.delta_pkr >= 0;
        const width = Math.max(4, (Math.abs(item.delta_pkr) / maxImpact) * 100);
        return `<div class="impact">
          <div class="impact-head"><span>${item.feature}</span>
          <span class="${up ? "up-text" : "down-text"}">${signedPKR(item.delta_pkr)}</span></div>
          <div class="bar-track"><div class="bar ${up ? "up" : "down"}" style="width:${width}%"></div></div>
        </div>`;
      })
      .join("");

    const whatIf = data.what_if;
    document.getElementById("what-if").textContent =
      `${whatIf.change}: ${formatPKR(whatIf.new_price_pkr)} ` +
      `(${signedPKR(whatIf.delta_pkr)} vs this estimate)`;

    document.getElementById("result").hidden = false;
    document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Estimate price";
  }
});

loadCityAverages();
