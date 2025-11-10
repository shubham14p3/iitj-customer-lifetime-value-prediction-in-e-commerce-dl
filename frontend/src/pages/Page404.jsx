// src/pages/Page404.tsx
import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Box, Typography, Button } from "@mui/material";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";

export default function Page404() {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate("/model/step1");
    }, 8000);
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <Box
      sx={{
        height: "100vh",
        width: "100%",
        background:
          "linear-gradient(135deg, #004d40 0%, #00695c 50%, #009688 100%)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        color: "#fff",
        textAlign: "center",
      }}
    >
      <ErrorOutlineIcon sx={{ fontSize: 80, mb: 2 }} />
      <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
        404 — Page Not Found
      </Typography>
      <Typography variant="body1" sx={{ opacity: 0.9, mb: 3 }}>
        Oops! The page you are looking for doesn’t exist. <br />
        Redirecting to Step 1 in 8 seconds...
      </Typography>
      <Button
        variant="contained"
        sx={{
          backgroundColor: "#00796b",
          "&:hover": { backgroundColor: "#004d40" },
        }}
        onClick={() => navigate("/model/step1")}
      >
        Go to Step 1
      </Button>
    </Box>
  );
}
