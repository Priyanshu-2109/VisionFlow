"use client";
import { CheckCircle } from "lucide-react";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

export default function WhyUs() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const benefits = [
    {
      title: "AI-Powered Insights",
      description: "Get real-time, data-driven insights to make smarter business decisions.",
    },
    {
      title: "Seamless Data Integration",
      description: "Easily connect and integrate your data from multiple sources without coding.",
    },
    {
      title: "Automated Workflows",
      description: "Streamline your operations with intelligent automation and workflow optimization.",
    },
    {
      title: "Scalable & Secure",
      description: "Built to handle enterprise-level scalability with top-tier security standards.",
    },
  ];

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

  return (
    <motion.section
      ref={ref}
      className="py-16 bg-[rgb(10,10,10)] text-white text-center"
      initial="hidden"
      id="why-us"
      animate={isInView ? "visible" : "hidden"}
      variants={containerVariants}
    >
      {/* Header */}
      <motion.div className="max-w-4xl mx-auto" variants={itemVariants}>
        <h2 className="text-4xl font-bold">Why Choose VisionFlow?</h2>
        <p className="text-lg text-gray-300 mt-4">
          Empower your business with AI-driven automation and seamless data integration.
        </p>
      </motion.div>

      {/* Benefits Grid */}
      <motion.div
        className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto mt-12"
        variants={containerVariants}
      >
        {benefits.map((benefit, index) => (
          <motion.div
            key={index}
            className="flex items-start space-x-4 p-6 bg-gray-800 rounded-xl shadow-md"
            variants={itemVariants}
            // whileHover={{ scale: 1.05 }}
          >
            <CheckCircle className="text-green-500 w-8 h-8 flex-shrink-0" />
            <div>
              <h3 className="text-xl font-semibold">{benefit.title}</h3>
              <p className="text-gray-400 mt-2">{benefit.description}</p>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </motion.section>
  );
}
