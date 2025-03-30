"use client";
import { CheckCircle } from "lucide-react";

export default function WhyUs() {
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

  return (
    <section className="py-20 bg-rgb(10,10,10) text-white text-center">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-4xl font-bold">Why Choose VisionFlow?</h2>
        <p className="text-lg text-gray-300 mt-4">
          Empower your business with AI-driven automation and seamless data integration.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto mt-12">
        {benefits.map((benefit, index) => (
          <div key={index} className="flex items-start space-x-4 p-6 bg-gray-800 rounded-xl shadow-md">
            <CheckCircle className="text-green-500 w-8 h-8 flex-shrink-0" />
            <div>
              <h3 className="text-xl font-semibold">{benefit.title}</h3>
              <p className="text-gray-400 mt-2">{benefit.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
