// src/pages/ModelStep1_LoadModel.tsx
import React, { useState } from "react";
import {
  Box,
  Paper,
  Typography,
  Stack,
  Button,
  CircularProgress,
  Alert,
  Divider,
  Card,
  CardContent,
  Tooltip,
  Fade,
  useMediaQuery,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import {
  loadClvModel,
  getClvModelInfo,
  downloadModel,
  triggerFileDownload,
} from "../lib/api";

export default function ModelStep1_LoadModel() {
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  const [modelFile, setModelFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [info, setInfo] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);

  // -------------------------
  // Submit to backend
  // -------------------------
  async function handleSubmit() {
    if (!modelFile) {
      setError("Please select a .pth model file or use the default model.");
      return;
    }
    try {
      setSubmitting(true);
      setError(null);
      const res = await loadClvModel(modelFile);
      if (!res.success) {
        setModelLoaded(false);
        setMessage(res.message || "Failed to load model.");
        return;
      }
      setModelLoaded(true);
      setMessage(res.message || "Model loaded successfully.");
      const infoRes = await getClvModelInfo();
      if (infoRes.success && infoRes.info) setInfo(infoRes.info);
    } catch (e: any) {
      setError(e?.message || "Error loading model.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDownloadModel() {
    try {
      const blob = await downloadModel();
      triggerFileDownload(blob, "saved_model.pth");
    } catch {
      setError("No model available for download or backend not ready.");
    }
  }

  // -------------------------
  // UI
  // -------------------------
  return (
    <Box
      sx={{
        position: "relative",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        minHeight: "calc(100vh - 160px)",
        px: { xs: 1, sm: 2 },
      }}
    >
      {/* Loader Overlay */}
      {submitting && (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            bgcolor: "rgba(255,255,255,0.6)",
            zIndex: 10,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: 3,
          }}
        >
          <CircularProgress size={50} />
        </Box>
      )}

      <Box sx={{ maxWidth: 800, width: "100%" }}>
        <Typography
          variant={isMobile ? "h6" : "h5"}
          sx={{ mb: 2, fontWeight: 600, textAlign: "center" }}
        >
          🧩 Step 1 – Load Model
        </Typography>

        {/* Info / Error Messages */}
        {error && (
          <Alert severity="error" sx={{ mb: 2, borderRadius: 2 }}>
            {error}
          </Alert>
        )}
        {message && !error && (
          <Alert severity={modelLoaded ? "success" : "info"} sx={{ mb: 2 }}>
            {message}
          </Alert>
        )}

        {/* Upload / Submit Section */}
        <Paper
          sx={{
            p: { xs: 2, sm: 3 },
            borderRadius: 3,
            mb: 3,
            textAlign: "center",
          }}
        >
          <Stack spacing={2} alignItems="center">
            <Typography variant="body1">
              Upload a trained <b>.pth</b> PyTorch model or use the default one
              on the server.
            </Typography>

            <Tooltip
              title={
                <Box sx={{ maxWidth: 260 }}>
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                    💡 Expected Model Type
                  </Typography>
                  <Typography variant="body2">
                    Feed-forward or LSTM network trained on CLV features
                    (≈16 inputs). File must be a PyTorch checkpoint such as{" "}
                    <code>saved_model.pth</code>.
                  </Typography>
                </Box>
              }
              placement="top"
              TransitionComponent={Fade}
              PopperProps={{
                modifiers: [
                  {
                    name: "preventOverflow",
                    options: { boundary: "viewport" },
                  },
                ],
              }}
            >
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontStyle: "italic" }}
              >
                Hover to see model requirements
              </Typography>
            </Tooltip>

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
                sx={{
                  textTransform: "none",
                  borderRadius: 2,
                  minWidth: 180,
                }}
              >
                Choose Model (.pth)
                <input
                  type="file"
                  accept=".pth"
                  hidden
                  onChange={(e) => setModelFile(e.target.files?.[0] || null)}
                />
              </Button>

              <Button
                variant="contained"
                onClick={handleSubmit}
                disabled={submitting}
                sx={{
                  textTransform: "none",
                  borderRadius: 2,
                  minWidth: 160,
                }}
              >
                {submitting ? (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={18} color="inherit" />
                    <span>Uploading…</span>
                  </Stack>
                ) : (
                  "Submit"
                )}
              </Button>

              <Button
                variant="outlined"
                color="secondary"
                onClick={handleDownloadModel}
                sx={{
                  textTransform: "none",
                  borderRadius: 2,
                  minWidth: 200,
                }}
              >
                📥 Download Current Model
              </Button>
            </Stack>

            {modelFile && (
              <Typography variant="body2" color="text.secondary">
                Selected file: <b>{modelFile.name}</b>
              </Typography>
            )}
          </Stack>
        </Paper>

        {/* Interactive Model Info Card */}
        {modelLoaded && info && (
          <Card
            sx={{
              borderRadius: 4,
              overflow: "hidden",
              mb: 4,
              boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
            }}
          >
            {/* Header */}
            <Box
              sx={{
                background:
                  "linear-gradient(90deg, #00796b 0%, #004d40 100%)",
                color: "white",
                px: 3,
                py: 1.5,
                textAlign: "center",
              }}
            >
              <Typography variant="h6" fontWeight={600}>
                ✅ Model Loaded Successfully
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.85 }}>
                Your CLV model is ready for predictions.
              </Typography>
            </Box>

            {/* Content */}
            <CardContent
              sx={{
                backgroundColor: "#fafafa",
                p: { xs: 2, sm: 3 },
              }}
            >
              <Stack spacing={1.5} alignItems={isMobile ? "center" : "flex-start"}>
                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  spacing={2}
                  justifyContent="center"
                  alignItems="center"
                >
                  <Typography variant="body1">
                    <b>🧠 Type:</b> {info.model_type}
                  </Typography>
                  <Typography variant="body1">
                    <b>📏 Input Dim:</b> {info.input_dim}
                  </Typography>
                </Stack>

                <Typography variant="body1">
                  <b>🧩 Hidden Layers:</b>{" "}
                  <span style={{ color: "#00695c", fontWeight: 500 }}>
                    {info.hidden_dims?.join(", ") || "N/A"}
                  </span>
                </Typography>

                <Stack
                  direction={{ xs: "column", sm: "row" }}
                  spacing={2}
                  justifyContent="center"
                  alignItems="center"
                >
                  <Typography variant="body1">
                    <b>💧 Dropout:</b> {info.dropout_rate ?? "N/A"}
                  </Typography>
                  <Typography variant="body1">
                    <b>🔢 Parameters:</b> {info.total_parameters}
                  </Typography>
                  <Typography variant="body1">
                    <b>⚙️ Trainable:</b> {info.trainable_parameters}
                  </Typography>
                </Stack>

                <Typography variant="body1">
                  <b>💻 Device:</b>{" "}
                  <span
                    style={{
                      color:
                        info.device === "cuda" ? "#007bff" : "#2e7d32",
                      fontWeight: 600,
                      textTransform: "uppercase",
                    }}
                  >
                    {info.device}
                  </span>
                </Typography>
              </Stack>
            </CardContent>
          </Card>
        )}

        {/* Next Step Button */}
        {modelLoaded && (
          <Box textAlign="center">
            <Button
              variant="contained"
              color="success"
              sx={{ borderRadius: 2, px: 4, py: 1, mt: 2 }}
              onClick={() => navigate("/model/step2")}
            >
              Next → Step 2 (Sample Data)
            </Button>
          </Box>
        )}
      </Box>
    </Box>
  );
}
