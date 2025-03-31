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
      className="py-24 bg-gradient-to-b bg-[rgb(10,10,10)] text-white"
      initial="hidden"
      id="why-us"
      animate={isInView ? "visible" : "hidden"}
      variants={containerVariants}
    >
      <div className="max-w-7xl mx-auto px-6">
        {/* Header with accent border */}
        <motion.div 
          className="max-w-4xl mx-auto text-center relative pb-4 mb-16"
          variants={itemVariants}
        >
          <div className="absolute left-1/2 transform -translate-x-1/2 -top-4 w-16 h-1 bg-gradient-to-r from-gray-700 to-gray-600 rounded-full"></div>
          <h2 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-200 to-gray-400">
            Why Choose VisionFlow?
          </h2>
          <p className="text-lg md:text-xl text-gray-400 mt-6 leading-relaxed">
            Empower your business with AI-driven automation and seamless data integration.
          </p>
          <div className="absolute left-1/2 transform -translate-x-1/2 bottom-0 w-24 h-1 bg-gradient-to-r from-gray-600 to-gray-700 rounded-full"></div>
        </motion.div>

        {/* Benefits Grid - Enhanced with better card design */}
        <motion.div
          className="grid md:grid-cols-2 gap-10 max-w-5xl mx-auto"
          variants={containerVariants}
        >
          {benefits.map((benefit, index) => (
            <motion.div
              key={index}
              className="group relative flex flex-col p-8 bg-gradient-to-br from-[rgb(18,18,22)] to-[rgb(15,15,18)] rounded-2xl shadow-xl border border-gray-800/60 hover:border-gray-700/60 transition-all duration-300 overflow-hidden"
              variants={itemVariants}
              whileHover={{ y: -5, transition: { duration: 0.2 } }}
            >
              {/* Subtle accent glow on hover */}
              <div className="absolute -top-20 -right-20 w-40 h-40 bg-gray-700/10 rounded-full filter blur-3xl opacity-0 group-hover:opacity-60 transition-opacity duration-700"></div>
              
              <div className="mb-5 flex items-center justify-center w-14 h-14 rounded-full bg-gradient-to-br from-gray-800 to-gray-900 border border-gray-700/50 shadow-lg">
                <CheckCircle className="text-gray-300 w-7 h-7" />
              </div>
              
              <h3 className="text-2xl font-bold mb-3 group-hover:text-white transition-colors">
                {benefit.title}
              </h3>
              
              <p className="text-gray-400 group-hover:text-gray-300 transition-colors">
                {benefit.description}
              </p>
              
              {/* Bottom accent line that animates on hover */}
              <div className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-gray-700 to-gray-600 group-hover:w-full transition-all duration-500"></div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </motion.section>
  );
}