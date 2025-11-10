// src/lib/api.ts
import axios from "axios";

export const BASE_URL =
  (import.meta as any).env?.VITE_API_URL || "http://13.50.9.79:5000";
  // (import.meta as any).env?.VITE_API_URL || "http://127.0.0.1:5000";

// -----------------------------
// Types for CLV Backend
// -----------------------------
export interface ClvSummary {
  total_customers: number;
  mean_predicted_clv: number;
  median_predicted_clv: number;
  min_predicted_clv: number;
  max_predicted_clv: number;
  std_predicted_clv: number;
}

export interface ClvMetrics {
  MSE: number;
  RMSE: number;
  MAE: number;
  R2: number;
  MAPE: number;
}

export interface PredictResponse {
  success: boolean;
  message?: string;
  predictions?: any[];
  total_rows?: number;
  metrics?: ClvMetrics | null;
  summary?: ClvSummary | null;
  plot?: string | null;
  download_path?: string | null;
}

export interface LoadModelResponse {
  success: boolean;
  message?: string;
  model_path?: string;
  device?: string;
}

export interface ModelInfoResponse {
  success: boolean;
  message?: string;
  info?: Record<string, any>;
}

// -----------------------------
// CLV Backend API Wrappers
// -----------------------------

export async function loadClvModel(
  modelFile?: File
): Promise<LoadModelResponse> {
  const formData = new FormData();
  if (modelFile) {
    formData.append("model_file", modelFile);
  }

  const res = await axios.post(`${BASE_URL}/load_model`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function getClvModelInfo(): Promise<ModelInfoResponse> {
  const res = await axios.get(`${BASE_URL}/model_info`);
  return res.data;
}

export async function predictClv(dataFile: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("data_file", dataFile);

  const res = await axios.post(`${BASE_URL}/predict`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export function getClvDownloadUrl(downloadPath: string): string {
  const filename = downloadPath.includes("/")
    ? downloadPath.split("/").pop()!
    : downloadPath;
  return `${BASE_URL}/download/${encodeURIComponent(filename)}`;
}

export async function downloadModel(): Promise<Blob> {
  const res = await axios.get(`${BASE_URL}/download_model`, {
    responseType: "blob",
  });
  return res.data;
}

export async function downloadSampleCsv(): Promise<Blob> {
  const res = await axios.get(`${BASE_URL}/download_sample`, {
    responseType: "blob",
  });
  return res.data;
}

export function triggerFileDownload(data: Blob, filename: string) {
  const url = window.URL.createObjectURL(data);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}
