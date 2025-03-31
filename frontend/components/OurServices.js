"use client";
import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { BarChart, Workflow, Database, ArrowRight } from "lucide-react";

const services = [
  {
    title: "AI-Powered Analytics",
    description:
      "Leverage advanced machine learning algorithms to transform your raw data into actionable business insights and predictive models.",
    icon: BarChart,
    color: "from-blue-500 to-purple-600",
  },
  {
    title: "Automated Workflows",
    description:
      "Eliminate repetitive tasks and streamline operations with intelligent automation that adapts to your business processes.",
    icon: Workflow,
    color: "from-green-500 to-teal-600",
  },
  {
    title: "Seamless Data Integration",
    description:
      "Connect and unify all your data sources without complex coding, enabling a complete view of your business operations.",
    icon: Database,
    color: "from-orange-500 to-red-600",
  },
];

const OurServices = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, staggerChildren: 0.1 }, // Faster entry animation
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3 } }, // Faster card appearance
  };

  return (
    <motion.section
      ref={ref}
      id="services"
      className="relative bg-[rgb(10,10,10)] text-white py-24 mb-20 px-6 md:px-12 lg:px-24 overflow-hidden"
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={containerVariants}
    >
      <div className="container mx-auto relative z-10">
        <div className="text-center mb-16">
          <motion.h2
            className="text-4xl md:text-5xl font-bold mb-4 text-white"
            variants={itemVariants}
          >
            Our Services
          </motion.h2>
          <motion.p
            className="max-w-2xl mx-auto text-gray-400 text-lg"
            variants={itemVariants}
          >
            Empowering your business with cutting-edge solutions that drive
            growth and efficiency
          </motion.p>
        </div>

        <motion.div
          className="grid md:grid-cols-3 gap-8 relative z-10"
          variants={containerVariants}
        >
          {services.map((service, index) => {
            const Icon = service.icon;
            return (
              <motion.div
                key={index}
                className="bg-gray-800/50 backdrop-blur-sm p-8 rounded-xl border border-gray-700 transition-all duration-500 ease-[cubic-bezier(0.25,0.1,0.25,1.0)] group hover:shadow-2xl hover:shadow-blue-900/10 hover:border-blue-500/50"
                variants={itemVariants}
                whileHover={{
                  y: -10,
                  transition: {
                    y: {
                      type: "spring",
                      stiffness: 200, // Increased stiffness for faster movement
                      damping: 10, // Lower damping for quicker return
                    },
                    duration: 0.4, // Faster hover effect
                  },
                }}
              >
                <div className="relative mb-6">
                  <div
                    className={`w-16 h-16 rounded-lg flex items-center justify-center bg-gradient-to-br ${service.color} shadow-lg transform group-hover:scale-105 transition-all duration-500 ease-out relative z-10`}
                  >
                    <Icon className="text-white w-8 h-8" />
                  </div>
                </div>

                <h3 className="text-2xl font-bold mb-3 transition-colors duration-300 group-hover:text-blue-400">
                  {service.title}
                </h3>

                <p className="text-gray-400 mb-6 leading-relaxed">
                  {service.description}
                </p>

                <div className="h-0.5 w-0 bg-gradient-to-r from-blue-400 to-blue-600 group-hover:w-16 transition-all duration-400 ease-out mb-4"></div>

                <a
                  href="#"
                  className="inline-flex items-center text-blue-400 group-hover:text-blue-300 transition-all duration-400"
                >
                  Learn more
                  <span className="relative ml-2 overflow-hidden inline-block">
                    <ArrowRight className="w-4 h-4 transform translate-x-0 group-hover:translate-x-5 transition-transform duration-400 ease-out" />
                    <ArrowRight className="w-4 h-4 absolute top-0 left-0 transform -translate-x-5 group-hover:translate-x-0 transition-transform duration-400 ease-out" />
                  </span>
                </a>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </motion.section>
  );
};

export default OurServices;
