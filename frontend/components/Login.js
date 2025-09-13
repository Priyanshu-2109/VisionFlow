"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { FcGoogle } from "react-icons/fc";
import axios from "axios";

export default function Login({ onClose }) {
  const router = useRouter();
  const [state, setState] = useState("Login");
  const [username, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [phone, setPhone] = useState(""); // Added phone state
  const [error, setError] = useState("");

  const backendUrl =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    
    try {
      if (state === "Login") {
        const response = await axios.post(`${backendUrl}/login`, {
          email,
          password,
        });

        if (response.data.token) {
          localStorage.setItem("token", response.data.token);
          localStorage.setItem("user", JSON.stringify(response.data));
          onClose();
          router.push("/");
        }
      } else {
        const response = await axios.post(`${backendUrl}/register`, {
          username,
          email,
          password,
          phone, // Added phone to the registration payload
        });

        if (response.data.token) {
          localStorage.setItem("token", response.data.token);
          localStorage.setItem("user", JSON.stringify(response.data));
          onClose();
          router.push("/");
        }
      }
    } catch (err) {
      setError(err.response?.data?.message || "An error occurred");
      console.log(err);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      const width = 500;
      const height = 600;
      const left = window.screen.width / 2 - width / 2;
      const top = window.screen.height / 2 - height / 2;

      const popup = window.open(
        `${backendUrl}/auth/google/callback`,
        "Google Login",
        `width=${width},height=${height},left=${left},top=${top}`
      );

      window.addEventListener("message", async (event) => {
        if (event.origin === "{backendUrl}") {
          if (event.data.success) {
            localStorage.setItem("token", event.data.token);
            localStorage.setItem("user", JSON.stringify(event.data.user));
            onClose();
            router.push("/");
          } else {
            setError(event.data.message || "Google login failed");
          }
        }
      });
    } catch (error) {
      setError("Failed to initiate Google login");
    }
  };

  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "unset";
    };
  }, []);

  const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.3 } },
    exit: { opacity: 0, transition: { duration: 0.3 } },
  };

  const formVariants = {
    hidden: { opacity: 0, y: -50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: "easeOut" },
    },
    exit: { opacity: 0, y: -50, transition: { duration: 0.5, ease: "easeIn" } },
  };

  return (
    <motion.div
      className="fixed inset-0 z-[9999] font-semibold flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
      variants={backdropVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
    >
      <motion.form
        onClick={(e) => e.stopPropagation()}
        className="bg-[rgb(15,15,15)] p-8 rounded-lg shadow-2xl w-96 text-white relative border border-gray-800"
        variants={formVariants}
        initial="hidden"
        animate="visible"
        exit="exit"
        onSubmit={handleSubmit}
      >
        <button
          type="button"
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
          onClick={onClose}
        >
          ✕
        </button>
        <div className="flex flex-col gap-2">
          <h1 className="text-2xl font-bold text-center text-white">
            {state === "Login" ? "Welcome back!" : "Welcome!"}
          </h1>
          <p className="text-sm text-gray-400 text-center mb-6">
            {state === "Login"
              ? "Please sign in to continue your journey"
              : "Create an account to get started"}
          </p>
        </div>

        {state === "Sign Up" && (
          <input
            type="text"
            placeholder="Full Name"
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg mb-4 text-white placeholder:text-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
            value={username}
            onChange={(e) => setName(e.target.value)}
            required
          />
        )}

        {/* Phone input field - only shown during sign up */}
        {state === "Sign Up" && (
          <input
            type="tel"
            placeholder="Phone Number"
            className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg mb-4 text-white placeholder:text-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            required
          />
        )}

        <input
          type="email"
          placeholder="Email Address"
          className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg mb-4 text-white placeholder:text-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg mb-4 text-white placeholder:text-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {state === "Login" && (
          <p className="text-sm text-blue-400 hover:text-blue-300 text-right cursor-pointer mb-4 transition-colors">
            Forgot password?
          </p>
        )}

        {error && (
          <div className="text-red-500 text-sm text-center mb-4">{error}</div>
        )}

        <button
          type="submit"
          className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-3 text-md font-medium rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-300 shadow-md"
        >
          {state === "Login" ? "Login" : "Sign Up"}
        </button>

        {/* Separator Line */}
        <div className="flex items-center gap-2 my-6">
          <div className="w-full h-px bg-gray-700"></div>
          <span className="text-gray-500 text-sm whitespace-nowrap">OR</span>
          <div className="w-full h-px bg-gray-700"></div>
        </div>

        {/* Google Login Button */}
        <button
          type="button"
          onClick={handleGoogleLogin}
          className="w-full flex items-center justify-center gap-2 border border-gray-700 text-white py-3 rounded-lg mb-4 hover:bg-gray-800 transition-colors"
        >
          <FcGoogle size={22} />
          <span>Continue with Google</span>
        </button>

        <p className="text-center text-gray-400 text-sm mt-6">
          {state === "Login" ? (
            <>
              Don&apos;t have an account?{" "}
              <span
                className="text-blue-400 hover:text-blue-300 cursor-pointer transition-colors"
                onClick={() => setState("Sign Up")}
              >
                Sign Up
              </span>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <span
                className="text-blue-400 hover:text-blue-300 cursor-pointer transition-colors"
                onClick={() => setState("Login")}
              >
                Login
              </span>
            </>
          )}
        </p>
      </motion.form>
    </motion.div>
  );
}
