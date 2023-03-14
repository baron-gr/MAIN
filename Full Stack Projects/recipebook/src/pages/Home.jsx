import Veggie from "../components/Veggie";
import Popular from "../components/Popular";
import MyRecipes from "../components/MyRecipes";
import { motion } from "framer-motion";

import React from 'react'

function Home() {
  return (
    <motion.div
    animate={{opacity: 1}}
    initial={{opacity: 0}}
    exit={{opacity: 0}}
    transition={{duration: 0.6}}
    >
      <Popular />
      <Veggie />
    </motion.div>
  )
}

export default Home