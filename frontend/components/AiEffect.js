"use client";
import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";
import Orb from "./ui/Orb"; // Import Orb component

const AiEffect = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  // Animation Variants
  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, staggerChildren: 0.2 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  };

  const orbVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { opacity: 1, scale: 1, transition: { duration: 0.6 } },
  };

  return (
    <motion.section
      ref={ref}
      id="ai"
      className="bg-[rgb(10,10,10)] text-white min-h-screen flex flex-col md:flex-row items-center justify-between px-12 py-20"
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={containerVariants}
    >
      {/* Left Side: Orb Animation */}
      <motion.div
        className="w-full md:w-1/2 flex justify-center"
        variants={orbVariants}
      >
        <div style={{ width: "100%", height: "600px", position: "relative" }}>
          <Orb
            hoverIntensity={0.5}
            rotateOnHover={true}
            hue={0}
            forceHoverState={false}
          />
        </div>
      </motion.div>

      {/* Right Side: AI-Powered Business Info */}
      <motion.div
        className="w-full md:w-1/2 text-center md:text-left"
        variants={itemVariants}
      >
        <h1 className="text-5xl font-bold leading-tight">
          AI-Powered Business Transformation
        </h1>
        <p className="text-lg text-gray-400 mt-4">
          Artificial Intelligence is redefining how businesses operate by
          optimizing workflows, automating tasks, and providing intelligent
          insights.
        </p>
        <p className="text-lg text-gray-400 mt-4">
          From predictive analytics to customer personalization, AI helps
          companies enhance efficiency, reduce costs, and drive innovation.
        </p>
      </motion.div>
    </motion.section>
  );
};

export default AiEffect;
