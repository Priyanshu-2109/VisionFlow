import express from "express";
import cors from "cors";
import 'dotenv/config'
import cookieParser from "cookie-parser";
import session from "express-session";
import connectDB from "./config/db.js";
import passport from "./config/passport.js";
import userRouter from "./routes/user.routes.js";

// dotenv.config({ path: "./.env" });

const app = express();
const port = process.env.PORT || 5000;

// Middleware
app.use(
  cors({
    origin: process.env.FRONTEND_URL || process.env.CORS_ORIGIN,
    credentials: true,
  })
);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());
app.use(
  session({
    secret: process.env.SESSION_SECRET || "1234",
    resave: false,
    saveUninitialized: false,
  })
);
app.use(passport.initialize());
app.use(passport.session());

// Routes
app.use("/api/auth", userRouter);

app.get("/", (req, res) => {
  res.send("API Working...");
});

// Database connection and server start
connectDB()
  .then(() => {
    app.on("error", (err) => {
      console.log(`Error: ${err}`);
      throw err;
    });

    app.listen(port, () => {
      console.log(`Server running on http://localhost:${port}`);
    });
  })
  .catch((err) => {
    console.log(`MongoDB connection failed: ${err}`);
  });

export default app ;
