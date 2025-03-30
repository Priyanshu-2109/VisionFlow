"use client";
import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";

const BusinessImpact = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  // Animation Variants
  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  };

  return (
    <motion.section
      ref={ref}
      id="effects"
      className="bg-[rgb(10,10,10)] text-white mb-10 flex items-center px-12"
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={containerVariants}
    >
      {/* Left Side: Text Content */}
      <motion.div className="w-1/2" variants={itemVariants}>
        <h1 className="text-5xl font-bold">
          How Does It Affect Your Business?
        </h1>
        <p className="text-lg text-gray-400 mt-6">
          Discover how AI-driven automation and smart data insights can optimize
          your operations, enhance customer engagement, and drive significant
          revenue growth for your business.
        </p>
      </motion.div>

      {/* Right Side: Lottie Animation */}
      <motion.div
        className="w-1/2 flex justify-center"
        variants={itemVariants}
      >
        <DotLottieReact
          src="https://lottie.host/ec82dfe2-93a6-4f89-8b75-09d2ccff0a8c/knin2IXWur.lottie"
          loop
          autoplay
          style={{ width: "100%", maxWidth: "700px", height: "auto" }}
        />
      </motion.div>
    </motion.section>
  );
};

export default BusinessImpact;