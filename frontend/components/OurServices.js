"use client";
import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { BarChart, Workflow, Database } from "lucide-react";

const services = [
  { title: "AI-Powered Analytics", description: "Gain deep insights with AI.", icon: BarChart },
  { title: "Automated Workflows", description: "Streamline operations.", icon: Workflow },
  { title: "Seamless Data Integration", description: "Unify your data easily.", icon: Database },
];

const OurServices = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

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
      id="services"
      className="relative bg-[rgb(10,10,10)] text-white py-16 px-6 md:px-12 lg:px-24 mt-15 mb-25"
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={containerVariants}
    >
      {/* Decorative Background */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="w-96 h-96 bg-gradient-to-r from-green-400 to-blue-500 rounded-full blur-3xl absolute top-10 left-10"></div>
        <div className="w-96 h-96 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full blur-3xl absolute bottom-10 right-10"></div>
      </div>

      {/* Heading */}
      <motion.h2
        className="text-4xl font-bold text-center mb-12 bg-clip-text text-white bg-gradient-to-r from-green-400 to-blue-500"
        variants={itemVariants}
      >
        Our Services
      </motion.h2>

      {/* Services Grid */}
      <motion.div className="grid md:grid-cols-3 gap-8 relative z-10" variants={containerVariants}>
        {services.map((service, index) => {
          const Icon = service.icon;
          return (
            <motion.div
              key={index}
              className="p-6 border border-gray-700 rounded-lg text-center hover:bg-transperent transition transform"
              variants={itemVariants}
              whileHover={{ scale: 1.05 }} // No green shadow effect
            >
              <Icon className="text-green-400 w-12 h-12 mx-auto mb-4" />
              <h3 className="text-xl font-semibold">{service.title}</h3>
              <p className="text-gray-400 mt-2">{service.description}</p>
            </motion.div>
          );
        })}
      </motion.div>
    </motion.section>
  );
};

export default OurServices;
