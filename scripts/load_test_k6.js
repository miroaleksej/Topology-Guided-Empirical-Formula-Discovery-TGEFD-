import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: __ENV.VUS ? parseInt(__ENV.VUS, 10) : 5,
  duration: __ENV.DURATION || "2m",
};

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.API_KEY || "";

const payload = JSON.stringify({
  tgefd_config_version: "1.0",
  run: { mode: "discover", seed: 1, deterministic: true },
  dataset: {
    source: "inline",
    x: { format: "array", values: [[0.0], [0.1], [0.2], [0.3]] },
    y: { values: [0.0, 0.1, 0.2, 0.3] },
  },
  hypothesis_space: {
    features: [
      { name: "const", expression: "1" },
      { name: "x", expression: "x" },
      { name: "x_pow", expression: "x^p", parameters: ["p"] },
    ],
    parameters: { p: { type: "float", values: [1.0, 2.0] } },
    regularization: { lambda: [0.01] },
  },
  topology: {
    persistent_homology: { max_dim: 1, metric: "euclidean" },
    persistence_image: {
      birth_range: [0.0, 1.0],
      pers_range: [0.0, 1.0],
      pixel_size: 0.05,
      weight: { type: "persistence", params: { n: 2.0 } },
      kernel: { type: "gaussian", params: { sigma: [[0.05, 0.0], [0.0, 0.05]] } },
    },
    stability_threshold: 0.2,
  },
  noise: { enabled: true, type: "gaussian", levels: [0.0], trials_per_level: 1 },
  evaluation: {
    acceptance: { min_stable_components: 1, require_h1: false },
    stability_score: { method: "pi_energy", aggregation: "mean", tolerance: 0.1 },
    rejection_reasons: ["no_stable_components"],
  },
  output: {
    verbosity: "summary",
    save_artifacts: false,
    formats: ["json"],
    paths: { base_dir: "./runs" },
  },
  reproducibility: {
    tgefd_version: "1.0.0",
    library_versions: true,
    hash_algorithm: "sha256",
  },
});

export default function () {
  const headers = {
    "Content-Type": "application/json",
  };
  if (API_KEY) {
    headers["x-api-key"] = API_KEY;
  }
  const res = http.post(`${BASE_URL}/v1/discover`, payload, { headers });
  check(res, {
    "status is 200/422": (r) => r.status === 200 || r.status === 422,
  });
  sleep(1);
}
