"use client";
import React from "react";

export default function HeroSection() {
    return (
      <section className="bg-rgb(10,10,10) text-white min-h-screen flex pt-20 items-center px-12">
        <div className="max-w-3xl">
          <h1 className="text-6xl font-bold leading-tight">
            Your fastest path to production
          </h1>
          <p className="text-xl text-gray-400 mt-6">
          Empower your business with our CRM tool - streamline customer interactions, boost sales, and drive growth effortlessly.
          </p>
          <div className="mt-8 flex space-x-4">
            <button className="bg-white text-black px-6 py-3 text-lg font-medium rounded-md flex items-center">
              Get Started for Free →
            </button>
            <button className="border border-gray-500 text-white px-6 py-3 text-lg font-medium rounded-md flex items-center">
              Contact Sales →
            </button>
          </div>
        </div>
      </section>
    );
  }
  