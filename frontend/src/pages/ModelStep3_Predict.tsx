import React, { useMemo, useState, useEffect, useRef } from "react";
import {
    Box,
    Paper,
    Typography,
    Stack,
    Button,
    CircularProgress,
    Alert,
} from "@mui/material";
import { DataGrid, GridColDef } from "@mui/x-data-grid";
import {
    predictClv,
    getClvDownloadUrl,
    ClvSummary,
    ClvMetrics,
} from "../lib/api";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import InsightsIcon from "@mui/icons-material/Insights";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import DownloadIcon from "@mui/icons-material/Download";
import LogoutIcon from "@mui/icons-material/Logout";
import html2canvas from "html2canvas";
import { useNavigate } from "react-router-dom";
import Tooltip from "@mui/material/Tooltip";
export default function ModelStep3_Predict() {
    const [dataFile, setDataFile] = useState<File | null>(null);
    const [predicting, setPredicting] = useState(false);
    const [summary, setSummary] = useState<ClvSummary | null>(null);
    const [metrics, setMetrics] = useState<ClvMetrics | null>(null);
    const [plotBase64, setPlotBase64] = useState<string | null>(null);
    const [rows, setRows] = useState<any[]>([]);
    const [downloadPath, setDownloadPath] = useState<string | null>(null);
    const [totalRows, setTotalRows] = useState<number | null>(null);
    const [error, setError] = useState<string | null>(null);

    const resultRef = useRef<HTMLDivElement>(null);
    const navigate = useNavigate();

    // ✅ Check if Step 2 already uploaded data
    useEffect(() => {
        const uploaded = localStorage.getItem("clv_last_upload");
        if (uploaded) {
            handlePredict(false);
        }
    }, []);

    // ✅ Save new upload to localStorage for persistence
    function handleFileChange(file: File | null) {
        setDataFile(file);
        if (file) localStorage.setItem("clv_last_upload", file.name);
    }

    // ✅ Download chart or full result as PNG
    async function handleDownloadScreenshot() {
        if (!resultRef.current) return;
        const canvas = await html2canvas(resultRef.current, {
            backgroundColor: "#ffffff",
            scale: 2,
        });
        const link = document.createElement("a");
        link.download = "clv_results.png";
        link.href = canvas.toDataURL("image/png");
        link.click();
    }

    // ✅ Logout and clear session
    function handleLogout() {
        localStorage.removeItem("ev_auth");
        localStorage.removeItem("ev_user");
        localStorage.removeItem("clv_last_upload");
        navigate("/login");
    }

    // ✅ Predict logic
    async function handlePredict(fromUpload = true) {
        try {
            setPredicting(true);
            setError(null);

            let fileToSend = dataFile ?? undefined;
            if (!fileToSend && fromUpload) {
                setError("Please upload a CSV file first.");
                setPredicting(false);
                return;
            }

            const res = await predictClv(fileToSend as any);
            if (!res.success) {
                setError(res.message || "Backend could not process the file.");
                return;
            }

            setSummary(res.summary || null);
            setMetrics((res.metrics as ClvMetrics | null) || null);
            setPlotBase64(res.plot || null);

            const preds = res.predictions || [];
            const sample = preds.slice(0, 30).map((r: any, i: number) => ({
                id: i + 1,
                ...r,
            }));
            setRows(sample);
            setTotalRows(res.total_rows ?? preds.length);
            setDownloadPath(res.download_path || null);
        } catch (e: any) {
            setError(e?.message || "Error connecting to backend.");
        } finally {
            setPredicting(false);
        }
    }

    // ✅ Table columns
    const cols: GridColDef[] = useMemo(() => {
        if (!rows.length) return [];
        return Object.keys(rows[0])
            .filter((k) => k !== "id")
            .map((k) => ({
                field: k,
                headerName: k
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (c) => c.toUpperCase()),
                flex: 1,
                minWidth: 150,
                headerAlign: "center",
                align: "center",
            }));
    }, [rows]);

    const downloadUrl = downloadPath && getClvDownloadUrl(downloadPath);

    // ✅ UI
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
                    📤 Step 3 – Predict CLV
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
                    {predicting && (
                        <Stack
                            direction="row"
                            justifyContent="center"
                            alignItems="center"
                            spacing={1}
                            sx={{ py: 3 }}
                        >
                            <CircularProgress size={28} />
                            <Typography variant="body1">
                                Generating predictions… please wait
                            </Typography>
                        </Stack>
                    )}

                    {/* If no cached data */}
                    {!predicting &&
                        !summary &&
                        !rows.length &&
                        !metrics &&
                        !plotBase64 && (
                            <Stack spacing={2} alignItems="center">
                                <Typography variant="body1" textAlign="center">
                                    No data found from Step 2.<br />
                                    Please upload your CSV again to generate predictions.
                                </Typography>
                                <Button
                                    variant="outlined"
                                    component="label"
                                    startIcon={<UploadFileIcon />}
                                    sx={{
                                        textTransform: "none",
                                        borderRadius: 2,
                                        minWidth: 200,
                                    }}
                                >
                                    Upload CSV File
                                    <input
                                        type="file"
                                        accept=".csv"
                                        hidden
                                        onChange={(e) =>
                                            handleFileChange(e.target.files?.[0] || null)
                                        }
                                    />
                                </Button>

                                {dataFile && (
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        disabled={predicting}
                                        onClick={() => handlePredict(true)}
                                        sx={{
                                            textTransform: "none",
                                            borderRadius: 2,
                                            px: 4,
                                            py: 1,
                                        }}
                                    >
                                        🚀 Generate Predictions
                                    </Button>
                                )}
                            </Stack>
                        )}

                    {/* Results Section */}
                    {!predicting && (summary || metrics || rows.length > 0) && (
                        <Box ref={resultRef}>
                            <Stack spacing={3}>
                                {summary && (
                                    <Box
                                        sx={{
                                            background: "#004d40",
                                            color: "#fff",
                                            borderRadius: 2,
                                            px: 3,
                                            py: 1.5,
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
                                                ✅ Prediction Summary
                                            </Typography>
                                            <Stack direction="row" spacing={1} alignItems="center">
                                                <InsightsIcon fontSize="small" />
                                                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                                                    Customers: {summary.total_customers} | Mean CLV:{" "}
                                                    {summary.mean_predicted_clv.toFixed(2)} | Std:{" "}
                                                    {summary.std_predicted_clv.toFixed(2)}
                                                </Typography>
                                            </Stack>
                                        </Stack>
                                    </Box>
                                )}
                                {/* {metrics && (
  <Box
    sx={{
      textAlign: "center",
      background: "linear-gradient(90deg, #e0f2f1 0%, #ffffff 100%)",
      borderRadius: 3,
      px: 3,
      py: 2,
      boxShadow: "0 3px 12px rgba(0,0,0,0.08)",
      border: "1px solid #b2dfdb",
    }}
  >
    <Typography
      variant="h6"
      sx={{
        fontWeight: 700,
        color: "#004d40",
        mb: 1,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        gap: 1,
      }}
    >
      🧮 Model Performance
    </Typography>

    <Stack
      direction={{ xs: "column", sm: "row" }}
      justifyContent="center"
      alignItems="center"
      spacing={{ xs: 1, sm: 3 }}
      flexWrap="wrap"
    >
      <Typography variant="body1" sx={{ color: "#00695c" }}>
        <b>MSE:</b> {metrics.MSE.toFixed(4)}
      </Typography>
      <Typography variant="body1" sx={{ color: "#00695c" }}>
        <b>RMSE:</b> {metrics.RMSE.toFixed(4)}
      </Typography>
      <Typography variant="body1" sx={{ color: "#00695c" }}>
        <b>MAE:</b> {metrics.MAE.toFixed(4)}
      </Typography>
      <Typography variant="body1" sx={{ color: "#00695c" }}>
        <b>MAPE:</b> {metrics.MAPE.toFixed(2)}%
      </Typography>
      <Typography variant="body1" sx={{ color: "#00695c" }}>
        <b>R²:</b> {metrics.R2.toFixed(4)}
      </Typography>
    </Stack>
  </Box>
)} */}
                                {metrics && (
                                    <Box
                                        sx={{
                                            textAlign: "center",
                                            background: "linear-gradient(90deg, #e0f2f1 0%, #ffffff 100%)",
                                            borderRadius: 3,
                                            px: 3,
                                            py: 2.5,
                                            boxShadow: "0 3px 12px rgba(0,0,0,0.08)",
                                            border: "1px solid #b2dfdb",
                                            mt: 2,
                                        }}
                                    >
                                        <Typography
                                            variant="h6"
                                            sx={{
                                                fontWeight: 700,
                                                color: "#004d40",
                                                mb: 2,
                                                display: "flex",
                                                justifyContent: "center",
                                                alignItems: "center",
                                                gap: 1,
                                            }}
                                        >
                                            🧮 Model Performance
                                        </Typography>

                                        <Stack
                                            direction={{ xs: "column", sm: "row" }}
                                            justifyContent="center"
                                            alignItems="center"
                                            spacing={{ xs: 1.2, sm: 3 }}
                                            flexWrap="wrap"
                                        >
                                            <Tooltip
                                                title="Mean Squared Error – the average of the squares of the errors between predicted and actual CLV values. Lower is better."
                                                arrow
                                                placement="top"
                                            >
                                                <Typography variant="body1" sx={{ color: "#00695c", cursor: "help" }}>
                                                    <b>MSE:</b> {metrics.MSE.toFixed(4)}
                                                </Typography>
                                            </Tooltip>

                                            <Tooltip
                                                title="Root Mean Squared Error – the square root of MSE, representing the standard deviation of prediction errors."
                                                arrow
                                                placement="top"
                                            >
                                                <Typography variant="body1" sx={{ color: "#00695c", cursor: "help" }}>
                                                    <b>RMSE:</b> {metrics.RMSE.toFixed(4)}
                                                </Typography>
                                            </Tooltip>

                                            <Tooltip
                                                title="Mean Absolute Error – the average magnitude of errors between predicted and actual CLV, without considering direction."
                                                arrow
                                                placement="top"
                                            >
                                                <Typography variant="body1" sx={{ color: "#00695c", cursor: "help" }}>
                                                    <b>MAE:</b> {metrics.MAE.toFixed(4)}
                                                </Typography>
                                            </Tooltip>

                                            <Tooltip
                                                title="Mean Absolute Percentage Error – the average percentage difference between predicted and actual CLV values. Lower % is better."
                                                arrow
                                                placement="top"
                                            >
                                                <Typography variant="body1" sx={{ color: "#00695c", cursor: "help" }}>
                                                    <b>MAPE:</b> {metrics.MAPE.toFixed(2)}%
                                                </Typography>
                                            </Tooltip>

                                            <Tooltip
                                                title="R² Score – represents how well predictions approximate actual values (1.0 = perfect fit)."
                                                arrow
                                                placement="top"
                                            >
                                                <Typography variant="body1" sx={{ color: "#00695c", cursor: "help" }}>
                                                    <b>R²:</b> {metrics.R2.toFixed(4)}
                                                </Typography>
                                            </Tooltip>
                                        </Stack>
                                    </Box>
                                )}



                                {plotBase64 && (
                                    <Box>
                                        <Typography variant="subtitle1" gutterBottom>
                                            <ShowChartIcon
                                                sx={{
                                                    mr: 1,
                                                    verticalAlign: "middle",
                                                    color: "#00695c",
                                                }}
                                            />
                                            Visualization
                                        </Typography>
                                        <Box
                                            sx={{
                                                borderRadius: 2,
                                                overflow: "hidden",
                                                border: "1px solid #e0e0e0",
                                                bgcolor: "#fafafa",
                                                display: "flex",
                                                justifyContent: "center",
                                            }}
                                        >
                                            <img
                                                src={`data:image/png;base64,${plotBase64}`}
                                                alt="CLV plot"
                                                style={{
                                                    width: "90%",
                                                    maxHeight: "65vh",
                                                    objectFit: "contain",
                                                    display: "block",
                                                }}
                                            />
                                        </Box>
                                    </Box>
                                )}

                                {rows.length > 0 && (
                                    <Box>
                                        <Typography variant="subtitle1" gutterBottom>
                                            🔍 Sample Predictions ({rows.length} of {totalRows})
                                        </Typography>
                                        <DataGrid
                                            rows={rows}
                                            columns={cols.map((col) => ({
                                                ...col,
                                                renderCell: (params) => (
                                                    <Box
                                                        sx={{
                                                            whiteSpace: "nowrap",
                                                            overflow: "hidden",
                                                            textOverflow: "ellipsis",
                                                            textAlign: "center",
                                                        }}
                                                        title={String(params.value ?? "")}
                                                    >
                                                        {typeof params.value === "number"
                                                            ? params.value.toFixed(4)
                                                            : params.value}
                                                    </Box>
                                                ),
                                            }))}
                                            disableRowSelectionOnClick
                                            density="compact"
                                            sx={{
                                                height: "70vh",
                                                border: "1px solid #004d40",
                                                "& .MuiDataGrid-columnHeaders": {
                                                    backgroundColor: "#ffffff", // ✅ plain white background
                                                    color: "#000000", // ✅ black text
                                                    fontWeight: 700,
                                                    borderBottom: "2px solid #000000",
                                                },
                                                "& .MuiDataGrid-row:nth-of-type(odd)": {
                                                    backgroundColor: "#f9f9f9",
                                                },
                                                "& .MuiDataGrid-row:hover": {
                                                    backgroundColor: "#e0f2f1",
                                                },
                                                "& .MuiDataGrid-cell": {
                                                    textAlign: "center",
                                                },
                                            }}
                                        />
                                    </Box>
                                )}

                                {/* ✅ Bottom action bar */}
                                <Stack
                                    direction="row"
                                    spacing={2}
                                    justifyContent="center"
                                    sx={{ mt: 2 }}
                                >
                                    {downloadUrl && (
                                        <Button
                                            variant="outlined"
                                            href={downloadUrl}
                                            startIcon={<DownloadIcon />}
                                            sx={{ borderRadius: 2, textTransform: "none" }}
                                        >
                                            📥 Download Prediction CSV
                                        </Button>
                                    )}
                                    {plotBase64 && (
                                        <Button
                                            variant="outlined"
                                            color="success"
                                            startIcon={<DownloadIcon />}
                                            onClick={handleDownloadScreenshot}
                                            sx={{ borderRadius: 2, textTransform: "none" }}
                                        >
                                            📸 Download Chart / Page
                                        </Button>
                                    )}
                                    <Button
                                        variant="outlined"
                                        color="error"
                                        startIcon={<LogoutIcon />}
                                        onClick={handleLogout}
                                        sx={{ borderRadius: 2, textTransform: "none" }}
                                    >
                                        🚪 Logout
                                    </Button>
                                </Stack>
                            </Stack>
                        </Box>
                    )}
                </Paper>
            </Box>
        </Box>
    );
}
