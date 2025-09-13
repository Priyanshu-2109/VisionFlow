import { Router } from "express";
import { verifyJWT } from "../middleware/auth.middleware.js";
import {
  register,
  login,
  logout,
  verifyOtp,
  refreshAccessToken,
  googleAuth,
  sendOtpToEmail,
} from "../controllers/user.controller.js";

import passport from "../config/passport.js";

const router = Router();

router.route("/register").post(register);

router.route("/login").post(login);

router.route("/send-otp").post(sendOtpToEmail);

router.route("/verify-otp").post(verifyOtp);

router.route("/logout").post(verifyJWT, logout);

router.route("/refresh-token").post(verifyJWT, refreshAccessToken);

router
  .route("/auth/google")
  .get(passport.authenticate("google", { scope: ["profile", "email"] }));

router.route("/auth/google/callback").get(
  passport.authenticate("google", {
    failureRedirect: "/login",
    session: true,
  }),
  googleAuth
);

export default router;
