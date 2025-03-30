"use client";
import React from "react";
import Orb from "./ui/Orb"; // Import Orb component

const AiEffect = () => {
  return (
    <section className="bg-[rgb(10,10,10)] text-white min-h-screen flex flex-col md:flex-row items-center justify-between px-12 py-20">
      {/* Left Side: Orb Animation */}
      <div className="w-full md:w-1/2 flex justify-center">
        <div style={{ width: "100%", height: "600px", position: "relative" }}>
          <Orb hoverIntensity={0.5} rotateOnHover={true} hue={0} forceHoverState={false} />
        </div>
      </div>

      {/* Right Side: AI-Powered Business Info */}
      <div className="w-full md:w-1/2 text-center md:text-left">
        <h1 className="text-5xl font-bold leading-tight">
          AI-Powered Business Transformation
        </h1>
        <p className="text-lg text-gray-400 mt-4">
          Artificial Intelligence is redefining how businesses operate by optimizing workflows, automating tasks, and providing intelligent insights. 
        </p>
        <p className="text-lg text-gray-400 mt-4">
          From predictive analytics to customer personalization, AI helps companies enhance efficiency, reduce costs, and drive innovation.
        </p>
      </div>
    </section>
  );
};

export default AiEffect;
