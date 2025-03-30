"use client";
import React from "react";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";

const BusinessImpact = () => {
  return (
    <section className="bg-[rgb(10,10,10)] text-white mb-10 flex items-center px-12">
      {/* Left Side: Text Content */}
      <div className="w-1/2">
        <h1 className="text-5xl font-bold">
          How Does It Affect Your Business?
        </h1>
        <p className="text-lg text-gray-400 mt-6">
          Discover how AI-driven automation and smart data insights can optimize
          your operations, enhance customer engagement, and drive significant
          revenue growth for your business.
        </p>
      </div>

      {/* Right Side: Lottie Animation */}
      <div className="w-1/2 flex justify-center">
        <DotLottieReact
          src="https://lottie.host/ec82dfe2-93a6-4f89-8b75-09d2ccff0a8c/knin2IXWur.lottie"
          loop
          autoplay
          style={{ width: "100%", maxWidth: "700px", height: "auto" }} // Increased maxWidth
        />
      </div>
    </section>
  );
};

export default BusinessImpact;
