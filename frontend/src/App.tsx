// src/App.tsx
import React, { useEffect } from "react";
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Box,
  Link,
  Stack,
} from "@mui/material";
import GitHubIcon from "@mui/icons-material/GitHub";
import LinkedInIcon from "@mui/icons-material/LinkedIn";

import LoginPage from "./pages/LoginPage";
import ModelStep1_LoadModel from "./pages/ModelStep1_LoadModel";
import ModelStep2_SampleData from "./pages/ModelStep2_SampleData";
import ModelStep3_Predict from "./pages/ModelStep3_Predict";
import Page404 from "./pages/Page404";

function useAuth() {
  return !!localStorage.getItem("ev_auth");
}

function Protected({ children }: { children: React.ReactNode }) {
  const authed = useAuth();
  const loc = useLocation();
  if (!authed) return <Navigate to="/login" state={{ from: loc }} replace />;
  return <>{children}</>;
}

// -------------------------
// NavBar
// -------------------------
function NavBar() {
  const navigate = useNavigate();
  const authed = useAuth();
  const userInfo = localStorage.getItem("ev_user");
  const user = userInfo ? JSON.parse(userInfo) : null;

  return (
    <AppBar
      position="fixed"
      sx={{
        background: "linear-gradient(90deg, #00695c, #00897b)",
        zIndex: (theme) => theme.zIndex.drawer + 1,
        left: 0,
        right: 0,
        width: "100%",           // ✅ correct: fits container
        maxWidth: "100vw",       // ✅ prevents overflow on zoom
        boxSizing: "border-box",
        overflow: "hidden",      // ✅ ensure no bleed pixels
      }}
    >
      <Toolbar
        disableGutters
        sx={{
          gap: 1,
          flexWrap: "wrap",
          width: "100%",
          px: 2,                 // left/right padding (8px)
          mx: "auto",            // center content
          maxWidth: "100%",      // ✅ ensure stays inside width
        }}
      >
        <Typography
          variant="h6"
          sx={{ flexGrow: 1, cursor: "pointer", fontWeight: 600 }}
          onClick={() => navigate("/model/step1")}
        >
          ⚡Customer Lifetime Value (CLV) Modeling
        </Typography>

        {authed ? (
          <>
            <Button color="inherit" onClick={() => navigate("/model/step1")}>
              1 – Load Model
            </Button>
            <Button color="inherit" onClick={() => navigate("/model/step2")}>
              2 – Sample Data
            </Button>
            <Button color="inherit" onClick={() => navigate("/model/step3")}>
              3 – Predict CLV
            </Button>

            {user && (
              <Typography
                variant="body2"
                sx={{
                  mx: 2,
                  px: 1.5,
                  py: 0.3,
                  borderRadius: 1,
                  fontWeight: 500,
                  backgroundColor: "rgba(255,255,255,0.15)",
                  whiteSpace: "nowrap",
                }}
              >
                You are logged in as <b>{user.name}</b> ({user.id})
              </Typography>
            )}

            <Button
              color="inherit"
              onClick={() => {
                localStorage.removeItem("ev_auth");
                localStorage.removeItem("ev_user");
                navigate("/login");
              }}
            >
              Logout
            </Button>
          </>
        ) : (
          <Button color="inherit" onClick={() => navigate("/login")}>
            Login
          </Button>
        )}
      </Toolbar>
    </AppBar>
  );
}

// -------------------------
// Footer
// -------------------------
function Footer() {
  return (
    <AppBar
      position="fixed"
      sx={{
        top: "auto",
        bottom: 0,
        background: "linear-gradient(90deg, #004d40, #00695c)",
        color: "#fff",
        height: 56,
        justifyContent: "center",
        left: 0,
        right: 0,
        width: "100%",           // ✅ correct fit
        maxWidth: "100vw",       // ✅ no overflow at any zoom
        boxSizing: "border-box",
        overflow: "hidden",
      }}
    >
      <Toolbar
        disableGutters
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          width: "100%",
          px: 2,
          mx: "auto",
          maxWidth: "100%",
        }}
      >
        <Typography variant="body1" fontWeight={600}>
          ⚡Customer Lifetime Value (CLV) Modeling System
        </Typography>

        <Stack direction="row" spacing={3} alignItems="center">
          <Stack direction="row" spacing={1} alignItems="center">
            <Typography variant="body2">Shubham Raj (M24DE3076)</Typography>
            <Link
              href="https://github.com/shubham14p3"
              target="_blank"
              color="inherit"
            >
              <GitHubIcon fontSize="small" />
            </Link>
            <Link
              href="https://linkedin.com/in/shubham14p3"
              target="_blank"
              color="inherit"
            >
              <LinkedInIcon fontSize="small" />
            </Link>
          </Stack>
          <Typography variant="body2">Bhavesh Arora (M24DE3022)</Typography>
          <Typography variant="body2">Kanishka Dhindhwal (M24DE3043)</Typography>
          <Typography variant="body2">Jatin Shrivas (M24DE3039)</Typography>
        </Stack>
      </Toolbar>
    </AppBar>
  );
}


// -------------------------
// Main App
// -------------------------
export default function App() {
  const loc = useLocation();

  useEffect(() => {
    if (loc.pathname === "/login") {
      localStorage.removeItem("ev_auth");
      localStorage.removeItem("ev_user");
    }
  }, [loc.pathname]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        overflowX: "hidden", // ✅ prevents right gap
      }}
    >
      <NavBar />
      <Container
        maxWidth="xl"
        disableGutters
        sx={{
          flex: 1,
          py: 10,
          pb: 10,
          overflow: "auto",
        }}
      >
        <Routes>
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/model/step1"
            element={
              <Protected>
                <ModelStep1_LoadModel />
              </Protected>
            }
          />
          <Route
            path="/model/step2"
            element={
              <Protected>
                <ModelStep2_SampleData />
              </Protected>
            }
          />
          <Route
            path="/model/step3"
            element={
              <Protected>
                <ModelStep3_Predict />
              </Protected>
            }
          />
          <Route path="*" element={<Page404 />} />
        </Routes>
      </Container>
      <Footer />
    </Box>
  );
}
