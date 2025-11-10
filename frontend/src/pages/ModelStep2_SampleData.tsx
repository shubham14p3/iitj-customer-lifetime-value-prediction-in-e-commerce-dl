// src/pages/ModelStep2_SampleData.tsx
import React, { useState } from "react";
import {
  Box,
  Paper,
  Typography,
  Stack,
  Button,
  CircularProgress,
  Alert,
  Tooltip,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { DataGrid, GridColDef } from "@mui/x-data-grid";
import {
  downloadSampleCsv,
  triggerFileDownload,
  predictClv,
} from "../lib/api";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import BarChartIcon from "@mui/icons-material/BarChart";
import InsightsIcon from "@mui/icons-material/Insights";

// ---------------------------------------------------------------------
// Helper: convert technical column name → readable label
// ---------------------------------------------------------------------
function getReadableName(key: string): string {
  const mapping: Record<string, string> = {
    age: "Customer Age",
    avg_order_value: "Avg. Order Value ($)",
    avg_session_duration: "Avg. Session Duration (min)",
    bounce_rate: "Bounce Rate",
    category_diversity: "Category Diversity",
    clv: "Actual CLV ($)",
    predicted_clv: "Predicted CLV ($)",
    customer_segment: "Customer Segment",
    days_since_first_purchase: "Days Since 1st Purchase",
    days_since_last_purchase: "Days Since Last Purchase",
    email_clicks: "Email Clicks",
    email_opens: "Email Opens",
    gender: "Gender",
    premium_category_ratio: "Premium Category %",
    promo_code_usage: "Promo Code Usage",
    return_rate: "Return Rate",
    total_orders: "Total Orders",
    total_page_views: "Page Views",
  };
  return (
    mapping[key] ||
    key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  );
}

// ---------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------
export default function ModelStep2_SampleData() {
  const navigate = useNavigate();
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [previewRows, setPreviewRows] = useState<any[]>([]);
  const [columns, setColumns] = useState<GridColDef[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [downloadingSample, setDownloadingSample] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<any | null>(null);

  // Handle CSV upload (reset preview and errors)
  function handleFileChange(file: File | null) {
    setDataFile(file);
    setPreviewRows([]);
    setColumns([]);
    setSummary(null);
    setError(null);
  }

  // Download sample dataset from backend
  async function handleDownloadSample() {
    try {
      setDownloadingSample(true);
      setError(null);
      const blob = await downloadSampleCsv();
      triggerFileDownload(blob, "clv_features_sample.csv");
    } catch {
      setError("Sample data file not found on backend.");
    } finally {
      setDownloadingSample(false);
    }
  }

  // Submit CSV → backend /predict
  async function handleProceed() {
    if (!dataFile) {
      setError("Please upload a CSV file first.");
      return;
    }

    try {
      setSubmitting(true);
      setError(null);
      const res = await predictClv(dataFile);
      if (!res.success) {
        setError(res.message || "Backend error while processing file.");
        return;
      }

      // Build DataGrid
      if (res.predictions?.length) {
        const preds = res.predictions.slice(0, 50);
        const cols: GridColDef[] = Object.keys(preds[0] || {}).map((k) => ({
          field: k,
          headerName: getReadableName(k),
          flex: 1,
          minWidth: 160,
          headerAlign: "center",
          align: "center",
          renderHeader: (params) => (
            <Tooltip title={getReadableName(params.field)} arrow>
              <span>{getReadableName(params.field)}</span>
            </Tooltip>
          ),
        }));
        const rows = preds.map((r, i) => ({ id: i + 1, ...r }));
        setColumns(cols);
        setPreviewRows(rows);
      }

      setSummary(res.summary);
    } catch (e: any) {
      setError(e?.message || "Error connecting to backend.");
    } finally {
      setSubmitting(false);
    }
  }

  // ---------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        minHeight: "calc(100vh - 160px)",
        px: { xs: 1, sm: 3 },
      }}
    >
      <Box sx={{ width: "100%", maxWidth: "95vw" }}>
        <Typography
          variant="h5"
          sx={{ mb: 2, fontWeight: 600, textAlign: "center" }}
        >
          🧾 Step 2 – Sample Data
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2, borderRadius: 2 }}>
            ⚠ {error}
          </Alert>
        )}

        <Paper
          sx={{
            p: { xs: 2, sm: 3 },
            borderRadius: 3,
            mb: 4,
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
          }}
        >
          <Stack spacing={2} alignItems="center">
            <Typography
              variant="body1"
              sx={{ textAlign: "center", maxWidth: 720 }}
            >
              Upload your CSV file or download the sample dataset. <br />
              Once ready, click{" "}
              <b>“Proceed to Analyze Sample Data”</b> to send it to the backend.
            </Typography>

            {/* Upload / Download buttons */}
            <Stack
              direction={{ xs: "column", sm: "row" }}
              spacing={2}
              alignItems="center"
              justifyContent="center"
              flexWrap="wrap"
            >
              <Button
                variant="outlined"
                component="label"
                startIcon={<UploadFileIcon />}
                sx={{ textTransform: "none", borderRadius: 2, minWidth: 180 }}
              >
                Upload CSV File
                <input
                  type="file"
                  accept=".csv"
                  hidden
                  onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                />
              </Button>

              <Typography variant="body2" color="text.secondary">
                {dataFile ? (
                  <>
                    Selected file: <b>{dataFile.name}</b>
                  </>
                ) : (
                  "No file selected."
                )}
              </Typography>

              <Button
                variant="outlined"
                color="secondary"
                startIcon={<BarChartIcon />}
                onClick={handleDownloadSample}
                disabled={downloadingSample}
                sx={{ textTransform: "none", borderRadius: 2, minWidth: 200 }}
              >
                {downloadingSample ? (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={18} />
                    <span>Downloading…</span>
                  </Stack>
                ) : (
                  "📥 Download Sample CSV"
                )}
              </Button>
            </Stack>

            {/* Proceed */}
            {dataFile && (
              <Button
                variant="contained"
                color="primary"
                onClick={handleProceed}
                disabled={submitting}
                sx={{
                  mt: 2,
                  borderRadius: 2,
                  px: 4,
                  py: 1,
                  textTransform: "none",
                  fontWeight: 600,
                }}
              >
                {submitting ? (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={18} color="inherit" />
                    <span>Processing…</span>
                  </Stack>
                ) : (
                  "🚀 Proceed to Analyze Sample Data"
                )}
              </Button>
            )}

            {/* Summary Bar */}
            {summary && (
              <Box
                sx={{
                  width: "100%",
                  background: "linear-gradient(90deg,#004d40,#00695c)",
                  color: "#fff",
                  borderRadius: 2,
                  px: 3,
                  py: 1.5,
                  mt: 3,
                  boxShadow: "0 3px 10px rgba(0,0,0,0.2)",
                }}
              >
                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  justifyContent="space-between"
                  alignItems={{ xs: "flex-start", sm: "center" }}
                  spacing={1}
                >
                  <Typography variant="h6" fontWeight={600}>
                    ✅ Sample Data Processed Successfully
                  </Typography>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <InsightsIcon fontSize="small" />
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Customers: {summary.total_customers} | Mean CLV:{" "}
                      {summary.mean_predicted_clv?.toFixed(2)} | Std:{" "}
                      {summary.std_predicted_clv?.toFixed(2)}
                    </Typography>
                  </Stack>
                </Stack>
              </Box>
            )}

            {/* Data Preview Table */}
            {previewRows.length > 0 && (
              <Box
                sx={{
                  width: "100%",
                  mt: 3,
                  borderRadius: 3,
                  bgcolor: "#f5f7f6",
                  p: 0,
                  overflowX: "auto",
                }}
              >
                <Box sx={{ minWidth: "1000px" }}>
                  <DataGrid
                    rows={previewRows}
                    columns={columns}
                    disableRowSelectionOnClick
                    density="compact"
                    autoHeight={false}
                    sx={{
                      height: "70vh",
                      border: "none",
                      backgroundColor: "#fff",
                      "--DataGrid-overlayHeight": "0px",

                      // ✅ HEADER STYLING (visible green text)
                      "& .MuiDataGrid-columnHeaders": {
                        position: "sticky",
                        top: 0,
                        zIndex: 2,
                        background:
                          "linear-gradient(90deg, #003d33, #004d40, #00695c)",
                        borderBottom: "2px solid #004d40",
                      },
                      "& .MuiDataGrid-columnHeaderTitle": {
                        color: "#A5D6A7 !important",
                        fontWeight: 700,
                        fontSize: "0.95rem",
                        textAlign: "center",
                        textShadow: "0 0 4px rgba(0,0,0,0.4)",
                        whiteSpace: "normal",
                      },

                      // ✅ Alternate rows
                      "& .MuiDataGrid-row:nth-of-type(odd)": {
                        backgroundColor: "#f9f9f9",
                      },
                      "& .MuiDataGrid-row:hover": {
                        backgroundColor: "#e0f2f1",
                      },

                      // ✅ Cell styling
                      "& .MuiDataGrid-cell": {
                        borderBottom: "1px solid #e0e0e0",
                        fontSize: "0.875rem",
                        color: "#212121",
                        textAlign: "center",
                      },

                      // ✅ Scrollbar styling
                      "& .MuiDataGrid-virtualScroller": {
                        scrollbarWidth: "thin",
                        "&::-webkit-scrollbar": { width: 8, height: 8 },
                        "&::-webkit-scrollbar-thumb": {
                          backgroundColor: "#009688",
                          borderRadius: 3,
                        },
                      },
                    }}
                  />
                </Box>
              </Box>
            )}
          </Stack>
        </Paper>

        {/* ✅ Next Step */}
        {summary && (
          <Box textAlign="center" sx={{ mt: 3 }}>
            <Button
              variant="contained"
              color="success"
              sx={{
                borderRadius: 2,
                px: 4,
                py: 1,
                fontWeight: 600,
              }}
              onClick={() => navigate("/model/step3")}
            >
              Next → Step 3 (Predict CLV)
            </Button>
          </Box>
        )}
      </Box>
    </Box>
  );
}
