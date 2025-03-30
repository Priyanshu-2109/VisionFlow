"use client";
import React from "react";
import { motion } from "framer-motion";
import { CheckCircle, BarChart, Workflow, Database } from "lucide-react";

const services = [
  { title: "AI-Powered Analytics", description: "Gain deep insights with AI.", icon: BarChart },
  { title: "Automated Workflows", description: "Streamline operations.", icon: Workflow },
  { title: "Seamless Data Integration", description: "Unify your data easily.", icon: Database },
];

const OurServices = () => {
  return (
    <div className="relative bg-gradient-to-b bg-rgb(10,10,10) text-white py-16 px-6 md:px-12 lg:px-24 mb-20">
      {/* Decorative Background */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="w-96 h-96 bg-gradient-to-r from-green-400 to-blue-500 rounded-full blur-3xl absolute top-10 left-10"></div>
        <div className="w-96 h-96 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full blur-3xl absolute bottom-10 right-10"></div>
      </div>

      <h2 className="text-4xl font-bold text-white text-center mb-12 bg-clip-text bg-gradient-to-r from-green-400 to-blue-500">
        Our Services
      </h2>
      <div className="grid md:grid-cols-3 gap-8 relative z-10">
        {services.map((service, index) => {
          const Icon = service.icon;
          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.2 }}
              whileHover={{ scale: 1.05 }}
              className="p-6 border border-gray-700 rounded-lg text-center hover:bg-gray-800 transition transform"
            >
              <Icon className="text-green-400 w-12 h-12 mx-auto mb-4" />
              <h3 className="text-xl font-semibold">{service.title}</h3>
              <p className="text-gray-400 mt-2">{service.description}</p>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default OurServices;