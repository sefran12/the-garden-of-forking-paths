import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoClient:
    def __init__(self):
        self.client = AsyncIOMotorClient(os.getenv('MONGODB_URI'))
        self.db = self.client[os.getenv('MONGODB_DB_NAME')]
        self.saves_collection = self.db[os.getenv('MONGODB_SAVES_COLLECTION')]
        logger.info(f"Connected to MongoDB database: {os.getenv('MONGODB_DB_NAME')}")
        logger.info(f"Using collection: {os.getenv('MONGODB_SAVES_COLLECTION')}")

    async def save_metadata(self, metadata: dict) -> str:
        logger.info(f"Saving metadata: {metadata}")
        result = await self.saves_collection.insert_one(metadata)
        save_id = str(result.inserted_id)
        logger.info(f"Saved metadata with ID: {save_id}")
        return save_id

    async def load_metadata(self, save_id: str) -> dict:
        logger.info(f"Loading metadata for save ID: {save_id}")
        metadata = await self.saves_collection.find_one({'_id': ObjectId(save_id)})
        logger.info(f"Loaded metadata: {metadata}")
        return metadata

    async def update_metadata(self, save_id: str, metadata: dict) -> bool:
        logger.info(f"Updating metadata for save ID: {save_id}")
        result = await self.saves_collection.update_one(
            {'_id': ObjectId(save_id)},
            {'$set': metadata}
        )
        success = result.modified_count > 0
        logger.info(f"Update {'successful' if success else 'failed'} for save ID: {save_id}")
        return success

    async def delete_metadata(self, save_id: str) -> bool:
        logger.info(f"Deleting metadata for save ID: {save_id}")
        result = await self.saves_collection.delete_one({'_id': ObjectId(save_id)})
        success = result.deleted_count > 0
        logger.info(f"Deletion {'successful' if success else 'failed'} for save ID: {save_id}")
        return success

    async def list_saves(self) -> list:
        logger.info("Listing all saves")
        cursor = self.saves_collection.find()
        saves = await cursor.to_list(length=None)
        logger.info(f"Found {len(saves)} saves")
        return saves

mongo_client = MongoClient()